"""API FastAPI du moteur de détection de plagiat et de doublons."""

from contextlib import asynccontextmanager
from pathlib import Path
import html
import os
import re
import tempfile
import time

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / "frontend" / ".env")

from src.indexing.reference_index import (
    ensure_reference_index,
    load_reference_documents,
)
from src.passages.passage_engine import analyze_document_pair, adjusted_hybrid_score
from src.preprocessing.clean_text import clean_text
from src.similarity.model_manager import get_model, get_model_info


THRESHOLDS = {"plagiarism": 0.55, "duplicate": 0.70}
LEXICAL_WEIGHT = float(os.getenv("SIMILARITY_LEXICAL_WEIGHT", "0.30"))
SEMANTIC_WEIGHT = float(os.getenv("SIMILARITY_SEMANTIC_WEIGHT", "0.70"))
TOP_SOURCES = int(os.getenv("SIMILARITY_REFERENCE_TOP_K", "10"))
TOP_PASSAGE_SOURCES = int(os.getenv("SIMILARITY_PASSAGE_TOP_K", "3"))

if abs((LEXICAL_WEIGHT + SEMANTIC_WEIGHT) - 1.0) > 1e-6:
    raise RuntimeError("SIMILARITY_LEXICAL_WEIGHT + SIMILARITY_SEMANTIC_WEIGHT doit être égal à 1.0")

_REFERENCE_CACHE = {}


def read_document(path: Path) -> str:
    try:
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages if (page.extract_text() or "").strip())
        doc = Document(path)
    except Exception as exc:
        raise HTTPException(400, f"Document DOCX invalide : {exc}")

    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _build_reference_runtime_index(kind: str):
    cached = _REFERENCE_CACHE.get(kind)
    if cached is not None:
        return cached

    rows, embeddings = load_reference_documents(kind)
    texts = [row.cleaned_text for row in rows]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)

    cached = {
        "rows": rows,
        "embeddings": embeddings,
        "vectorizer": vectorizer,
        "tfidf_matrix": matrix,
    }
    _REFERENCE_CACHE[kind] = cached
    return cached


def _warm_reference_runtime_indexes() -> None:
    for kind in ("duplicate", "plagiarism"):
        try:
            _build_reference_runtime_index(kind)
        except Exception as exc:
            print(f"[similarity] Index {kind} non disponible : {exc}")


def _semantic_scores(candidate_embedding: np.ndarray, reference_embeddings: np.ndarray) -> np.ndarray:
    # Les embeddings sont normalisés : le produit scalaire est alors équivalent
    # à la similarité cosinus.
    scores = np.matmul(reference_embeddings, candidate_embedding)
    return np.clip(scores, 0.0, 1.0)


def score_documents(candidate_text: str, kind: str):
    if kind not in THRESHOLDS:
        raise HTTPException(404, "Mode inconnu. Utilisez plagiarism ou duplicate.")

    candidate = clean_text(candidate_text)
    if not candidate:
        return []

    index = _build_reference_runtime_index(kind)
    vectorizer = index["vectorizer"]
    reference_tfidf = index["tfidf_matrix"]
    reference_embeddings = index["embeddings"]
    rows = index["rows"]

    candidate_tfidf = vectorizer.transform([candidate])
    lexical = cosine_similarity(candidate_tfidf, reference_tfidf).ravel()

    candidate_embedding = get_model().encode(
        [candidate],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    semantic = _semantic_scores(candidate_embedding, reference_embeddings)

    hybrid = np.array(
        [
            adjusted_hybrid_score(
                candidate,
                rows[int(idx)].extracted_text or rows[int(idx)].cleaned_text,
                lexical[idx],
                semantic[idx],
                LEXICAL_WEIGHT,
                SEMANTIC_WEIGHT,
            )
            for idx in range(len(rows))
        ]
    )

    order = np.argsort(hybrid)[::-1]
    ranked = []

    for idx in order:
        row = rows[int(idx)]
        ranked.append(
            {
                "id": row.id,
                "source": Path(row.file_path).name or row.title,
                "title": row.title,
                "tfidf_score": round(float(lexical[idx]), 4),
                "semantic_score": round(float(semantic[idx]), 4),
                "hybrid_score": round(float(hybrid[idx]), 4),
                "_document": row,
            }
        )

    return ranked


def score_two_texts(candidate_text: str, source_text: str):
    candidate = clean_text(candidate_text)
    source = clean_text(source_text)
    if not candidate or not source:
        return None

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform([candidate, source])
    lexical = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0])

    embeddings = get_model().encode(
        [candidate, source],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    semantic = float(np.dot(embeddings[0], embeddings[1]))
    semantic = float(np.clip(semantic, 0.0, 1.0))
    hybrid = adjusted_hybrid_score(
        candidate,
        source,
        lexical,
        semantic,
        LEXICAL_WEIGHT,
        SEMANTIC_WEIGHT,
    )

    return {
        "tfidf_score": round(lexical, 4),
        "semantic_score": round(semantic, 4),
        "hybrid_score": round(hybrid, 4),
    }


def decision(score, threshold):
    return "SIMILAIRE" if score >= threshold else "DIFFERENT"


def run_detection(candidate_text: str, kind: str, threshold: float | None = None):
    candidate = clean_text(candidate_text)
    if not candidate:
        raise HTTPException(400, "Le document ne contient aucun texte exploitable.")

    ranked = score_documents(candidate, kind)
    if not ranked:
        raise HTTPException(
            500,
            f"Aucun document de référence indexé pour le mode « {kind} ».",
        )

    threshold = THRESHOLDS[kind] if threshold is None else float(threshold)
    if not 0.01 <= threshold <= 1.0:
        raise HTTPException(400, "Le seuil de similarité doit être compris entre 0.01 et 1.00.")
    best = ranked[0]
    passage_groups = []

    # L'analyse lourde des passages est limitée aux 3 sources les plus proches.
    for item in ranked[:TOP_PASSAGE_SOURCES]:
        document = item["_document"]
        source_text = document.extracted_text or document.cleaned_text
        detail = analyze_document_pair(
            candidate,
            source_text,
            threshold=0.50,
            top_k=8,
        )

        source_html = html.escape(source_text)
        candidate_html = html.escape(candidate)
        unique_passages = []
        for match in detail["matches"]:
            passage = match["source_passage"]
            if passage and passage not in unique_passages:
                unique_passages.append(passage)

        candidate_passages = []
        for match in detail["matches"]:
            passage = match["candidate_passage"]
            if passage and passage not in candidate_passages:
                candidate_passages.append(passage)

        for passage in sorted(candidate_passages, key=len, reverse=True):
            escaped_passage = html.escape(passage)
            candidate_html = candidate_html.replace(
                escaped_passage,
                f'<mark class="docsec-evidence">{escaped_passage}</mark>',
            )

        for passage in sorted(unique_passages, key=len, reverse=True):
            escaped_passage = html.escape(passage)
            source_html = source_html.replace(
                escaped_passage,
                f'<mark class="docsec-evidence">{escaped_passage}</mark>',
            )

        passage_groups.append(
            {
                "source_id": item["id"],
                "source": item["source"],
                "document_score": item["hybrid_score"],
                "tfidf_score": item["tfidf_score"],
                "semantic_score": item["semantic_score"],
                "coverage": detail["coverage"],
                "matches": detail["matches"],
                "passages_candidate": detail["passages_candidate"],
                "passages_source": detail["passages_source"],
                "source_document": source_text,
                "source_document_html": source_html,
                "candidate_document_html": candidate_html,
            }
        )

    return {
        "mode": kind,
        "decision": decision(best["hybrid_score"], threshold),
        "threshold": threshold,
        "lexical_weight": LEXICAL_WEIGHT,
        "semantic_weight": SEMANTIC_WEIGHT,
        "best_source": best["source"],
        "best_source_id": best["id"],
        "candidate_document_html": (
            passage_groups[0]["candidate_document_html"]
            if passage_groups else html.escape(candidate)
        ),
        "tfidf_score": best["tfidf_score"],
        "semantic_score": best["semantic_score"],
        "hybrid_score": best["hybrid_score"],
        "novelty_score": round(max(0.0, 1 - best["hybrid_score"]), 4),
        "sources": [
            {k: v for k, v in item.items() if k != "_document"}
            for item in ranked[:TOP_SOURCES]
        ],
        "matches": passage_groups,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Première exécution : charge/télécharge le modèle localement, applique les
    # migrations puis indexe le corpus dans PostgreSQL. Ensuite, les données
    # existantes sont réutilisées et aucun embedding du corpus n'est recalculé.
    model_info = get_model_info()
    print(f"[similarity] Modèle : {model_info['name']}")
    get_model()

    try:
        stats = ensure_reference_index()
        print(
            "[similarity] Références : "
            f"{stats['created_or_updated']} préparée(s), "
            f"{stats['skipped']} déjà indexée(s)."
        )
        _warm_reference_runtime_indexes()
    except Exception as exc:
        print(f"[similarity] Initialisation PostgreSQL/index impossible : {exc}")

    yield


app = FastAPI(
    title="TDRDOC-SCAN - Détection documentaire",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "tdrdoc-scan",
        "status": "ok",
        "version": "2.0.0",
    }


@app.get("/api/health")
def health():
    result = {
        "status": "ok",
        "model": get_model_info(),
        "reference_counts": {},
        "cache_ready": list(_REFERENCE_CACHE.keys()),
    }
    for kind in ("duplicate", "plagiarism"):
        try:
            result["reference_counts"][kind] = len(
                load_reference_documents(kind)[0]
            )
        except Exception:
            result["reference_counts"][kind] = 0
    return result


@app.post("/api/detect/{kind}")
async def detect(kind: str, file: UploadFile = File(...), threshold: float | None = Form(None)):
    if kind not in THRESHOLDS:
        raise HTTPException(
            404,
            "Mode inconnu. Utilisez plagiarism ou duplicate.",
        )
    if not file.filename or Path(file.filename).suffix.lower() not in {".docx", ".pdf"}:
        raise HTTPException(400, "Formats acceptés : DOCX et PDF.")

    start = time.perf_counter()
    safe_name = re.sub(r"[^\w.\- ]", "_", file.filename)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / safe_name
        path.write_bytes(await file.read())
        candidate_text = read_document(path)
        # run_detection utilise le Django ORM de façon synchrone.
        # On l'exécute dans un thread pour éviter SynchronousOnlyOperation
        # lorsque cet endpoint FastAPI est appelé depuis le contexte async.
        result = await run_in_threadpool(run_detection, candidate_text, kind, threshold)

    result["duration_ms"] = round((time.perf_counter() - start) * 1000)
    return result


@app.post("/api/compare")
async def compare(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as directory:
        a = Path(directory) / ("a" + Path(file_a.filename or "a.docx").suffix.lower())
        b = Path(directory) / ("b" + Path(file_b.filename or "b.docx").suffix.lower())
        a.write_bytes(await file_a.read())
        b.write_bytes(await file_b.read())
        result = score_two_texts(read_document(a), read_document(b))

    if result is None:
        raise HTTPException(400, "Les deux documents doivent contenir du texte.")
    return result

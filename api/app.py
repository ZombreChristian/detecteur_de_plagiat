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
from src.similarity.contextual_engine import analyze_context
from src.similarity.ai_judge import judge_documents, OLLAMA_ENABLED, OLLAMA_MODEL
from src.preprocessing.clean_text import clean_text
from src.similarity.model_manager import get_model, get_model_info


THRESHOLDS = {"plagiarism": 0.55, "duplicate": 0.70}
LEXICAL_WEIGHT = float(os.getenv("SIMILARITY_LEXICAL_WEIGHT", "0.30"))
SEMANTIC_WEIGHT = float(os.getenv("SIMILARITY_SEMANTIC_WEIGHT", "0.70"))
TOP_SOURCES = int(os.getenv("SIMILARITY_REFERENCE_TOP_K", "10"))
TOP_PASSAGE_SOURCES = int(os.getenv("SIMILARITY_PASSAGE_TOP_K", "5"))
AI_JUDGE_MAX_CALLS = int(os.getenv("AI_JUDGE_MAX_CALLS", "1"))

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

    # TF-IDF documentaire indépendant : il ne sert pas à décider si deux
    # passages sont sémantiquement similaires.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([candidate, source])
    document_lexical = float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0])

    detail = analyze_document_pair(
        candidate,
        source,
        top_k=8,
        lexical_weight=LEXICAL_WEIGHT,
        semantic_weight=SEMANTIC_WEIGHT,
        document_lexical_score=document_lexical,
    )
    return {
        "tfidf_score": detail["tfidf_score"],
        "semantic_score": detail["semantic_score"],
        "hybrid_score": detail["hybrid_score"],
        "coverage": detail["coverage"],
        "passages_similaires": len(detail["matches"]),
    }

def decision(score, threshold):
    return "SIMILAIRE" if score >= threshold else "DIFFERENT"


def _contextual_explanation(contextual):
    """Produit une explication courte et lisible de l'identité de l'étude."""
    dimensions = contextual.get("dimensions", {})
    labels = {
        "objet": "objet",
        "secteur": "secteur",
        "activites": "activités",
        "zone": "zone géographique",
        "objectifs": "objectifs",
        "resultats": "résultats attendus",
    }
    similar = [
        labels[name]
        for name, data in dimensions.items()
        if name in labels and data.get("status") == "SIMILAIRE"
    ]
    different = [
        labels[name]
        for name, data in dimensions.items()
        if name in labels and data.get("status") == "DIFFERENT"
    ]

    if contextual.get("verdict") == "SIMILAIRE":
        if similar:
            return (
                "Les deux documents présentent une identité d'étude fortement "
                f"concordante sur {', '.join(similar)}. "
                "Les différences de formulation ou de structure ne suffisent pas "
                "à remettre en cause cette proximité."
            )
        return "L'analyse contextuelle indique une forte proximité de l'identité de l'étude."

    if contextual.get("verdict") == "DIFFERENT":
        if different:
            return (
                "L'analyse contextuelle relève une divergence importante sur "
                f"{', '.join(different)}. La similarité de structure ou de formulation "
                "ne suffit donc pas à considérer les deux études comme identiques."
            )
        return "L'analyse contextuelle ne confirme pas une identité commune d'étude."

    return (
        "Les documents présentent des éléments communs, mais l'identité de l'étude "
        "reste insuffisamment établie pour une décision automatique."
    )



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

    # Le seuil choisi dans l'interface est uniquement un seuil de décision.
    # Le formulaire envoie un pourcentage (ex. 70) alors que le moteur compare
    # les scores sur une échelle 0-1 (ex. 0.70).
    if threshold is None:
        threshold = THRESHOLDS[kind]
    else:
        threshold = float(threshold)
        if threshold > 1.0:
            threshold /= 100.0

    if not 0.01 <= threshold <= 1.0:
        raise HTTPException(400, "Le seuil de décision doit être compris entre 1 % et 100 %.")
    best = ranked[0]
    passage_groups = []

    # Qwen vérifie uniquement le 3e résultat du classement.
    ai_target_id = ranked[2]["id"] if len(ranked) >= 3 else None
    ai_calls = 0

    # L'analyse lourde des passages est limitée aux sources les plus proches.
    for item in ranked[:TOP_PASSAGE_SOURCES]:
        document = item["_document"]
        source_text = document.extracted_text or document.cleaned_text
        detail = analyze_document_pair(
            candidate,
            source_text,
            top_k=8,
            lexical_weight=LEXICAL_WEIGHT,
            semantic_weight=SEMANTIC_WEIGHT,
            document_lexical_score=item["tfidf_score"],
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

        contextual = analyze_context(candidate, source_text)

        ai_judgment = None
        # Test volontaire : Qwen vérifie uniquement le 3e résultat.
        if (
            OLLAMA_ENABLED
            and item["id"] == ai_target_id
            and ai_calls < AI_JUDGE_MAX_CALLS
        ):
            ai_judgment = judge_documents(
                candidate,
                source_text,
                contextual_analysis=contextual,
                technical_evidence={
                    "rank": 3,
                    "tfidf_score": item["tfidf_score"],
                    "semantic_score": item["semantic_score"],
                    "hybrid_score": detail["hybrid_score"],
                    "coverage": detail["coverage"],
                },
            )
            if ai_judgment is not None:
                ai_calls += 1

        passage_groups.append(
            {
                "source_id": item["id"],
                "source": item["source"],
                # Ces trois scores proviennent désormais des correspondances
                # de passages, et non plus de la comparaison globale initiale.
                "document_score": detail["hybrid_score"],
                "tfidf_score": detail["tfidf_score"],
                "semantic_score": detail["semantic_score"],
                "hybrid_score": detail["hybrid_score"],
                "coverage": detail["coverage"],
                "matches": detail["matches"],
                "contextual_analysis": contextual,
                "contextual_verdict": contextual["verdict"],
                "contextual_confidence": contextual["confidence"],
                "ai_judgment": ai_judgment,
                "ai_verdict": ai_judgment.get("verdict") if ai_judgment else None,
                "ai_confidence": ai_judgment.get("confidence") if ai_judgment else None,
                "passages_candidate": detail["passages_candidate"],
                "passages_source": detail["passages_source"],
                "source_document": source_text,
                "source_document_html": source_html,
                "candidate_document_html": candidate_html,
            }
        )

    # La décision et les scores affichés utilisent une preuve mixte :
    # TF-IDF documentaire indépendant + sémantique issue des passages.
    # Le TF-IDF ne sert jamais de filtre pour accepter/refuser une correspondance
    # sémantique.
    evidence_groups = [group for group in passage_groups if group.get("matches")]
    # La preuve contextuelle intervient après les passages : elle permet à
    # l'IA d'écarter les faux rapprochements dus à la structure générique
    # des TDR, sans modifier les scores TF-IDF/sémantique calculés.
    # Le verdict Qwen du 3e résultat ne remplace pas le meilleur résultat global.
    evidence_groups.sort(
        key=lambda group: (
            group.get("contextual_verdict") == "SIMILAIRE",
            group.get("hybrid_score", 0.0),
        ),
        reverse=True,
    )
    best_evidence = evidence_groups[0] if evidence_groups else None

    if best_evidence is not None:
        score_decision = decision(best_evidence["hybrid_score"], threshold)
        contextual = best_evidence.get("contextual_analysis", {})
        context_verdict = best_evidence.get("contextual_verdict", "À EXAMINER")
        ai_judgment = best_evidence.get("ai_judgment")
        ai_verdict = best_evidence.get("ai_verdict")

        # Si Ollama est disponible, son jugement global est l'arbitre final
        # pour cette paire. Les moteurs numériques restent des preuves.
        if ai_verdict == "SIMILAIRE":
            final_decision = "SIMILAIRE"
        elif ai_verdict == "DIFFERENT":
            final_decision = "DIFFERENT"
        elif ai_verdict == "A_EXAMINER":
            final_decision = "À EXAMINER"
        else:
            final_decision = None

        # Fallback complet vers la logique existante si le modèle local
        # n'est pas installé, arrêté ou ne renvoie pas un JSON exploitable.

        # Le seuil reste la porte d'entrée de la décision, mais l'identité
        # contextuelle peut maintenant expliquer ou invalider une proximité
        # purement textuelle. Une divergence contextuelle ambiguë ne transforme
        # plus automatiquement un score élevé en DIFFERENT : elle passe par
        # À EXAMINER. Seule une contradiction forte et explicite est bloquante.
        overlaps = contextual.get("specific_overlap", {})
        dims = contextual.get("dimensions", {})
        hard_conflict = (
            context_verdict == "DIFFERENT"
            and contextual.get("confidence") == "FORTE"
            and (
                any(
                    dims.get(name, {}).get("score") is not None
                    and dims.get(name, {}).get("score") < 0.45
                    for name in ("objet", "secteur", "zone")
                )
                or any(
                    dims.get(name, {}).get("score") is not None
                    and dims.get(name, {}).get("score") < 0.45
                    and overlaps.get(name) is not None
                    and overlaps.get(name) < 0.10
                    for name in ("objet", "secteur", "zone")
                )
            )
        )

        if final_decision is None:
            if score_decision == "DIFFERENT":
                final_decision = "DIFFERENT"
            elif hard_conflict:
                final_decision = "DIFFERENT"
            elif context_verdict == "À EXAMINER":
                final_decision = "À EXAMINER"
            elif context_verdict == "DIFFERENT":
                final_decision = "À EXAMINER"
            else:
                final_decision = "SIMILAIRE"

        result_tfidf = best_evidence["tfidf_score"]
        result_semantic = best_evidence["semantic_score"]
        result_hybrid = best_evidence["hybrid_score"]
        result_source = best_evidence["source"]
        result_source_id = best_evidence["source_id"]
    else:
        final_decision = "DIFFERENT"
        result_tfidf = 0.0
        result_semantic = 0.0
        result_hybrid = 0.0
        result_source = best["source"]
        result_source_id = best["id"]

    # Les sources présentées sont celles effectivement comparées par passages.
    # Leur ordre est celui de la preuve obtenue, pas celui du seul score global.
    displayed_sources = sorted(
        passage_groups,
        key=lambda group: group.get("hybrid_score", 0.0),
        reverse=True,
    )

    return {
        "mode": kind,
        "decision": final_decision,
        "threshold": threshold,
        "threshold_percent": round(threshold * 100, 2),
        "lexical_weight": LEXICAL_WEIGHT,
        "semantic_weight": SEMANTIC_WEIGHT,
        "best_source": result_source,
        "best_source_id": result_source_id,
        "candidate_document_html": (
            best_evidence["candidate_document_html"]
            if best_evidence is not None
            else (passage_groups[0]["candidate_document_html"] if passage_groups else html.escape(candidate))
        ),
        "tfidf_score": result_tfidf,
        "semantic_score": result_semantic,
        "hybrid_score": result_hybrid,
        "novelty_score": round(max(0.0, 1 - result_hybrid), 4),
        "contextual_analysis": (
            best_evidence["contextual_analysis"]
            if best_evidence is not None
            else None
        ),
        "contextual_explanation": (
            _contextual_explanation(best_evidence["contextual_analysis"])
            if best_evidence is not None
            else "Aucune analyse contextuelle disponible."
        ),
        "ai_judgment": (
            best_evidence.get("ai_judgment")
            if best_evidence is not None
            else None
        ),
        "ai_model": OLLAMA_MODEL if OLLAMA_ENABLED else None,
        "ai_enabled": OLLAMA_ENABLED,
        "sources": displayed_sources[:TOP_SOURCES],
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

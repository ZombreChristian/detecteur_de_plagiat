"""Moteur RAG autonome dédié à la vérification des TDR.

Il interroge directement toutes les références TDR déjà indexées dans PostgreSQL.
Aucun TOP_K global : les références sont parcourues par lots techniques, puis
Qwen synthétise les éléments avant de rendre le verdict final.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend"
import sys
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "frontend.settings")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("TDR_RAG_MODEL", os.getenv("OLLAMA_MODEL", "qwen3:8b"))
OLLAMA_TIMEOUT = max(60, int(os.getenv("TDR_RAG_TIMEOUT_SECONDS", "300")))
# Recherche rapide lexical + sémantique sur l'ensemble du corpus, puis arbitrage RAG
# uniquement sur les trois références les mieux classées.
RAG_TOP_K = 3
REFERENCE_CHARS = 3200
CANDIDATE_CHARS = 9000
OUTPUT_TOKENS = 1200

CASES = {
    "CAS_1_CORRESPONDANCE_FORTE",
    "CAS_2_CORRESPONDANCE_PARTIELLE",
    "CAS_3_MEME_OBJECTIF_ZONE_DIFFERENTE",
    "CAS_4_AUCUNE_ETUDE_PERTINENTE",
}


class IndependentTDRError(RuntimeError):
    pass


def _ollama_json(system_prompt: str, user_prompt: str, tokens: int = OUTPUT_TOKENS) -> dict:
    """Appelle Ollama avec reprises en cas de JSON invalide ou de génération tronquée."""
    last_error = None
    last_content = ""
    # Une première tentative normale, puis deux reprises avec davantage de tokens
    # et une consigne explicite de produire un objet JSON complet.
    for attempt in range(3):
        token_budget = min(max(tokens, 600) * (attempt + 1), 1800)
        retry_note = ""
        if attempt:
            retry_note = (
                "\n\nIMPORTANT — la réponse précédente était invalide ou incomplète. "
                "Recommence et renvoie un unique objet JSON valide, complet, sans Markdown, "
                "sans texte avant/après, avec toutes les chaînes correctement échappées. "
                "Reste concis et termine toutes les accolades et tous les guillemets."
            )
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "stream": False,
                    "format": "json",
                    "think": False,
                    "keep_alive": "5m",
                    "options": {"temperature": 0, "num_predict": token_budget},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt + retry_note},
                    ],
                },
                timeout=OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            last_content = (payload.get("message") or {}).get("content", "").strip()
            if not last_content:
                raise ValueError("Ollama a renvoyé une réponse vide.")
            parsed = json.loads(last_content)
            if not isinstance(parsed, dict):
                raise ValueError("La réponse JSON n'est pas un objet.")
            return parsed
        except Exception as exc:
            last_error = exc
    preview = re.sub(r"\\s+", " ", last_content)[:240]
    raise IndependentTDRError(
        f"Qwen ({OLLAMA_MODEL}) a renvoyé un JSON invalide après 3 tentatives. "
        f"Dernière erreur : {last_error}. Début de la réponse : {preview!r}"
    ) from last_error


def extract_tdr_document(path: Path) -> str:
    """Extraction autonome du TDR candidat, sans passer par l'ancien moteur."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join(
            (page.extract_text() or "").strip()
            for page in reader.pages
            if (page.extract_text() or "").strip()
        )
    if suffix == ".docx":
        from docx import Document
        document = Document(str(path))
        parts = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
        return "\n\n".join(parts)
    raise IndependentTDRError("Format non pris en charge. Utilisez un fichier DOCX ou PDF.")


def _excerpt(text: str, limit: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", (text or "")).strip()
    if len(text) <= limit:
        return text
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    priority = ("objet", "contexte", "objectif", "résultat", "zone", "population",
                "méthodologie", "activité", "livrable", "période", "justification")
    ranked = sorted(
        enumerate(paragraphs),
        key=lambda item: sum(word in item[1][:180].lower() for word in priority),
        reverse=True,
    )
    selected, used = set(), 0
    for idx, paragraph in ranked:
        cost = len(paragraph) + 2
        if used + cost <= limit:
            selected.add(idx)
            used += cost
    result = "\n\n".join(p for i, p in enumerate(paragraphs) if i in selected)
    return result or text[:limit]


def _load_all_indexed_tdr():
    import django
    django.setup()
    from detector.models import StudyDocument
    rows = list(
        StudyDocument.objects.filter(
            is_reference=True, document_type="TDR"
        ).exclude(cleaned_text="").exclude(embedding=[]).order_by("id")
    )
    return rows


def _rank_all_references(candidate: str, rows):
    """Classe tout le corpus avec TF-IDF lexical (30 %) et cosinus sémantique (70 %)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from src.similarity.model_manager import get_model

    reference_texts = [
        (row.cleaned_text or row.extracted_text or row.title or "").strip()
        for row in rows
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform([candidate] + reference_texts)
        lexical_scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    except ValueError:
        lexical_scores = np.zeros(len(rows), dtype=np.float32)

    model = get_model()
    candidate_vector = model.encode(
        [candidate], normalize_embeddings=True, show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    ranked = []
    for index, row in enumerate(rows):
        vector = np.asarray(row.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        semantic = float(np.dot(vector / norm, candidate_vector)) if norm else 0.0
        semantic = max(0.0, min(1.0, semantic))
        lexical = max(0.0, min(1.0, float(lexical_scores[index])))
        hybrid = 0.30 * lexical + 0.70 * semantic
        ranked.append((hybrid, lexical, semantic, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked

def analyze_tdr_independently(candidate_text: str) -> dict:
    """Classe le corpus, puis demande à Qwen un seul verdict fondé sur les 3 meilleurs TDR."""
    candidate = (candidate_text or "").strip()
    if len(candidate) < 80:
        raise IndependentTDRError("Le TDR contient trop peu de texte exploitable.")
    rows = _load_all_indexed_tdr()
    if not rows:
        raise IndependentTDRError(
            "Aucun TDR de référence n'est présent dans l'index PostgreSQL. "
            "Préparez l'index avant de lancer l'analyse."
        )

    # Le classement lexical + sémantique parcourt le corpus, mais ne sollicite pas
    # le LLM pour chaque référence. Seuls les trois meilleurs candidats passent au RAG.
    ranked = _rank_all_references(candidate, rows)
    if not ranked:
        raise IndependentTDRError("Aucune référence TDR exploitable n'a été trouvée.")
    selected = ranked[:RAG_TOP_K]
    candidate_excerpt = _excerpt(candidate, CANDIDATE_CHARS)

    reference_blocks = []
    for rank, (hybrid, lexical, semantic, row) in enumerate(selected, start=1):
        reference_blocks.append(
            f"CANDIDAT DE RECHERCHE {rank}\n"
            f"TITRE: {row.title}\nFICHIER: {Path(row.file_path).name}\n"
            f"SCORES DE RÉCUPÉRATION (indicatifs seulement): lexical={lexical:.3f}, "
            f"sémantique={semantic:.3f}, combiné={hybrid:.3f}\n"
            f"CONTENU DU TDR DE RÉFÉRENCE:\n"
            f"{_excerpt(row.extracted_text or row.cleaned_text, REFERENCE_CHARS)}"
        )

    system_prompt = (
        "Tu es un expert chargé de déterminer si une étude correspondant au TDR "
        "candidat a déjà été réalisée. Trois références ont été sélectionnées par "
        "recherche lexicale et sémantique dans la base. Analyse réellement le contenu "
        "des trois documents, leurs objectifs, objet, activités, zone, population, "
        "période, méthode et livrables. Les scores ne sont PAS des preuves ni des verdicts. "
        "Ne conclus pas qu'une étude existe simplement parce qu'un document est classé "
        "premier. Choisis un seul verdict final parmi : "
        "CAS_1_CORRESPONDANCE_FORTE (même étude ou quasi-identique), "
        "CAS_2_CORRESPONDANCE_PARTIELLE (recouvrement substantiel mais différence importante), "
        "CAS_3_MEME_OBJECTIF_ZONE_DIFFERENTE (étude comparable mais zone géographique différente), "
        "CAS_4_AUCUNE_ETUDE_PERTINENTE (aucune des références ne démontre une étude pertinente). "
        "Si aucune des trois références n'est réellement pertinente, choisis CAS_4. "
        "N'invente aucun fait. Ne donne PAS trois verdicts et ne présente PAS les trois "
        "documents comme trois résultats. Rends uniquement le verdict global et, si une "
        "référence soutient ce verdict, son nom exact. Réponds uniquement avec un objet "
        "JSON valide ayant les clés case_id, decision, explanation, matched_reference, "
        "confidence, common_points (tableau), differences (tableau), recommendation."
    )
    final = _ollama_json(
        system_prompt,
        "TDR CANDIDAT À ÉVALUER:\n" + candidate_excerpt
        + "\n\nLES TROIS RÉFÉRENCES À EXAMINER AVANT DE CONCLURE:\n"
        + "\n\n--- DOCUMENT SUIVANT ---\n\n".join(reference_blocks),
        tokens=1200,
    )

    case_id = str(final.get("case_id", "CAS_4_AUCUNE_ETUDE_PERTINENTE"))
    if case_id not in CASES:
        case_id = "CAS_4_AUCUNE_ETUDE_PERTINENTE"
    reference_name = str(final.get("matched_reference", "")).strip()
    allowed_names = {
        name
        for _, _, _, row in selected
        for name in (Path(row.file_path).name, row.title)
    }
    if reference_name not in allowed_names or case_id == "CAS_4_AUCUNE_ETUDE_PERTINENTE":
        reference_name = ""

    case_titles = {
        "CAS_1_CORRESPONDANCE_FORTE": "Correspondance forte — étude probablement déjà réalisée",
        "CAS_2_CORRESPONDANCE_PARTIELLE": "Correspondance partielle — vérification recommandée",
        "CAS_3_MEME_OBJECTIF_ZONE_DIFFERENTE": "Objectifs similaires — zone géographique différente",
        "CAS_4_AUCUNE_ETUDE_PERTINENTE": "Aucune étude pertinente trouvée dans la base",
    }
    matched_row = next(
        (row for _, _, _, row in selected
         if reference_name and reference_name in {Path(row.file_path).name, row.title}),
        None,
    )
    evidence = []
    if matched_row and case_id != "CAS_4_AUCUNE_ETUDE_PERTINENTE":
        evidence.append({
            "source": reference_name,
            "point": str(final.get("explanation", ""))[:900],
            "justification": "Référence retenue après examen du contenu par le RAG.",
        })

    return {
        "mode": "tdr_rag_independent",
        "decision_provider": "independent_rag",
        "decision": case_id,
        "rag_analysis": {
            "case_id": case_id,
            "case_title": case_titles[case_id],
            "model": OLLAMA_MODEL,
            "decision": str(final.get("decision", case_id)),
            "explanation": str(final.get("explanation", ""))[:2500],
            "matched_reference": reference_name,
            "confidence": str(final.get("confidence", "À confirmer"))[:80],
            "common_points": [str(x)[:500] for x in final.get("common_points", [])[:10]],
            "differences": [str(x)[:500] for x in final.get("differences", [])[:10]],
            "recommendation": str(final.get("recommendation", ""))[:2000],
            "references_examined": len(selected),
            "corpus_references_ranked": len(ranked),
            "batches_examined": 1,
            "evidence": evidence,
            # Pas de liste des trois candidats comme trois verdicts : un seul verdict global.
            "batch_findings": [],
        },
        "best_source_id": matched_row.id if matched_row else None,
        "best_source": reference_name or None,
        "sources": [
            {
                "source_id": matched_row.id,
                "source": Path(matched_row.file_path).name,
                "title": matched_row.title,
                "retrieval_score": round(
                    next(score for score, _, _, row in selected if row.id == matched_row.id), 4
                ),
            }
        ] if matched_row else [],
        "tfidf_score": None,
        "semantic_score": None,
        "hybrid_score": None,
        "novelty_score": None,
    }

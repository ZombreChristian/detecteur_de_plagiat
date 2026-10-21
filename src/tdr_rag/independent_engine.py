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
# Lot de traitement technique seulement : toutes les références sont parcourues.
BATCH_SIZE = 2
REFERENCE_CHARS = 1400
CANDIDATE_CHARS = 6500
OUTPUT_TOKENS = 400

CASES = {
    "CAS_1_CORRESPONDANCE_FORTE",
    "CAS_2_CORRESPONDANCE_PARTIELLE",
    "CAS_3_MEME_OBJECTIF_ZONE_DIFFERENTE",
    "CAS_4_AUCUNE_ETUDE_PERTINENTE",
}


class IndependentTDRError(RuntimeError):
    pass


def _ollama_json(system_prompt: str, user_prompt: str, tokens: int = OUTPUT_TOKENS) -> dict:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "format": "json",
                "think": False,
                "keep_alive": "5m",
                "options": {"temperature": 0.1, "num_predict": tokens},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        content = (payload.get("message") or {}).get("content", "")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("La réponse JSON n'est pas un objet.")
        return parsed
    except Exception as exc:
        raise IndependentTDRError(
            f"Le moteur RAG indépendant n'a pas pu obtenir une réponse exploitable de "
            f"{OLLAMA_MODEL} : {exc}"
        ) from exc


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
    from src.similarity.model_manager import get_model
    model = get_model()
    candidate_vector = model.encode(
        [candidate], normalize_embeddings=True, show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    ranked = []
    for row in rows:
        vector = np.asarray(row.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if not norm:
            continue
        score = float(np.dot(vector / norm, candidate_vector))
        ranked.append((score, row))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return ranked


def analyze_tdr_independently(candidate_text: str) -> dict:
    candidate = (candidate_text or "").strip()
    if len(candidate) < 80:
        raise IndependentTDRError("Le TDR contient trop peu de texte exploitable.")
    rows = _load_all_indexed_tdr()
    if not rows:
        raise IndependentTDRError(
            "Aucun TDR de référence n'est présent dans l'index PostgreSQL. "
            "Préparez l'index avant de lancer l'analyse."
        )

    ranked = _rank_all_references(candidate, rows)
    if not ranked:
        raise IndependentTDRError("Aucune référence TDR exploitable n'a été trouvée.")

    # Chaque document indexé est analysé, en lots de taille technique limitée.
    # Le nombre total de références n'est pas prédéfini ni plafonné.
    system_batch = (
        "Tu es un analyste expert des termes de référence. Compare le TDR candidat "
        "aux références fournies. Les scores de recherche ne sont pas des verdicts. "
        "Évalue l'identité réelle de l'étude : objet, problème, objectifs, activités, "
        "population, zone, période, méthode et livrables. N'invente rien. "
        "Réponds uniquement en JSON avec les clés: relevant (bool), findings (string), "
        "reference (string), similarities (array de chaînes), differences (array de chaînes)."
    )
    batch_findings = []
    candidate_excerpt = _excerpt(candidate, CANDIDATE_CHARS)
    for start in range(0, len(ranked), BATCH_SIZE):
        batch = ranked[start:start + BATCH_SIZE]
        reference_blocks = []
        for score, row in batch:
            reference_blocks.append(
                f"RÉFÉRENCE: {row.title}\nFICHIER: {Path(row.file_path).name}\n"
                f"PERTINENCE DE RECHERCHE (indicative): {score:.3f}\n"
                f"CONTENU:\n{_excerpt(row.extracted_text or row.cleaned_text, REFERENCE_CHARS)}"
            )
        analysis = _ollama_json(
            system_batch,
            "TDR CANDIDAT:\n" + candidate_excerpt + "\n\n"
            + "\n\n--- RÉFÉRENCE SUIVANTE ---\n\n".join(reference_blocks),
            tokens=320,
        )
        batch_findings.append({
            "reference": str(analysis.get("reference", ""))[:300],
            "relevant": bool(analysis.get("relevant", False)),
            "findings": str(analysis.get("findings", ""))[:900],
            "similarities": [str(x)[:300] for x in analysis.get("similarities", [])[:6]],
            "differences": [str(x)[:300] for x in analysis.get("differences", [])[:6]],
        })

    final_system = (
        "Tu es l'arbitre final d'une vérification de TDR. Les résultats fournis sont "
        "des analyses de lots couvrant toutes les références indexées. Décide selon "
        "les preuves, pas selon les scores. CAS_1 si même étude ou quasi-identique; "
        "CAS_2 si correspondance partielle substantielle mais différences importantes; "
        "CAS_3 si objectifs/étude comparables mais zone géographique différente; "
        "CAS_4 si aucune référence réellement pertinente. Si les preuves sont ambiguës, "
        "ne prétends pas à une certitude et explique les limites dans recommendation. "
        "Réponds uniquement en JSON: case_id, decision, explanation, matched_reference, "
        "confidence, common_points (array), differences (array), recommendation."
    )
    final = _ollama_json(
        final_system,
        "TDR CANDIDAT:\n" + candidate_excerpt + "\n\n"
        "RÉSULTATS DES COMPARAISONS AVEC TOUTES LES RÉFÉRENCES INDEXÉES:\n"
        + json.dumps(batch_findings, ensure_ascii=False),
        tokens=650,
    )
    case_id = str(final.get("case_id", "CAS_4_AUCUNE_ETUDE_PERTINENTE"))
    if case_id not in CASES:
        case_id = "CAS_4_AUCUNE_ETUDE_PERTINENTE"
    reference_name = str(final.get("matched_reference", "")).strip()
    allowed_names = {Path(row.file_path).name for _, row in ranked} | {row.title for _, row in ranked}
    if reference_name not in allowed_names:
        reference_name = ""
    return {
        "mode": "tdr_rag_independent",
        "decision_provider": "independent_rag",
        "decision": case_id,
        "rag_analysis": {
            "case_id": case_id,
            "decision": str(final.get("decision", case_id)),
            "explanation": str(final.get("explanation", ""))[:2500],
            "matched_reference": reference_name,
            "confidence": str(final.get("confidence", "À confirmer"))[:80],
            "common_points": [str(x)[:500] for x in final.get("common_points", [])[:10]],
            "differences": [str(x)[:500] for x in final.get("differences", [])[:10]],
            "recommendation": str(final.get("recommendation", ""))[:2000],
            "references_examined": len(ranked),
            "batches_examined": len(batch_findings),
            "batch_findings": batch_findings,
        },
        "sources": [
            {"source": Path(row.file_path).name, "title": row.title, "retrieval_score": round(score, 4)}
            for score, row in ranked if Path(row.file_path).name == reference_name or row.title == reference_name
        ],
        "tfidf_score": None,
        "semantic_score": None,
        "hybrid_score": None,
        "novelty_score": None,
    }

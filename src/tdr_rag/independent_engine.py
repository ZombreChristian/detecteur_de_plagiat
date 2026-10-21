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
            tokens=700,
        )
        batch_findings.append({
            "reference": str(analysis.get("reference", ""))[:300],
            "relevant": bool(analysis.get("relevant", False)),
            "findings": str(analysis.get("findings", ""))[:900],
            "similarities": [str(x)[:300] for x in analysis.get("similarities", [])[:6]],
            "differences": [str(x)[:300] for x in analysis.get("differences", [])[:6]],
        })

    # Réduction hiérarchique des résultats pour garder le contexte borné,
    # sans supprimer de lots ni plafonner le nombre de documents comparés.
    reduction_system = (
        "Synthétise fidèlement les constats fournis sur des comparaisons de TDR. "
        "Conserve les noms exacts des références, les similitudes substantielles, "
        "les différences de zone et les incertitudes. N'émets pas le verdict final. "
        "Réponds en JSON avec une clé summary contenant une synthèse concise."
    )
    summaries = [json.dumps(item, ensure_ascii=False) for item in batch_findings]
    while len(summaries) > 6:
        reduced = []
        for pos in range(0, len(summaries), 6):
            group = summaries[pos:pos + 6]
            summary = _ollama_json(
                reduction_system,
                "Synthétise ces constats sans inventer et conserve les noms de références :\n"
                + "\n\n".join(group),
                tokens=600,
            )
            reduced.append(str(summary.get("summary", ""))[:1800])
        summaries = reduced

    final_system = (
        "Tu es l'arbitre final d'une vérification de TDR. Les résultats fournis "
        "synthétisent les comparaisons de toutes les références indexées. Décide selon "
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
        "SYNTHÈSE DES COMPARAISONS AVEC TOUTES LES RÉFÉRENCES INDEXÉES:\n"
        + "\n\n".join(summaries),
        tokens=1000,
    )
    case_id = str(final.get("case_id", "CAS_4_AUCUNE_ETUDE_PERTINENTE"))
    if case_id not in CASES:
        case_id = "CAS_4_AUCUNE_ETUDE_PERTINENTE"
    reference_name = str(final.get("matched_reference", "")).strip()
    allowed_names = {Path(row.file_path).name for _, row in ranked} | {row.title for _, row in ranked}
    if reference_name not in allowed_names or case_id == "CAS_4_AUCUNE_ETUDE_PERTINENTE":
        reference_name = ""
    case_titles = {
        "CAS_1_CORRESPONDANCE_FORTE": "Correspondance forte — étude probablement déjà réalisée",
        "CAS_2_CORRESPONDANCE_PARTIELLE": "Correspondance partielle — vérification recommandée",
        "CAS_3_MEME_OBJECTIF_ZONE_DIFFERENTE": "Objectifs similaires — zone géographique différente",
        "CAS_4_AUCUNE_ETUDE_PERTINENTE": "Aucune étude pertinente trouvée dans la base",
    }
    matched_row = next(
        (row for _, row in ranked
         if reference_name and reference_name in {Path(row.file_path).name, row.title}),
        None,
    )
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
            "references_examined": len(ranked),
            "batches_examined": len(batch_findings),
            "evidence": [
                {"source": item["reference"], "point": item["findings"], "justification": ""}
                for item in batch_findings if item["relevant"] and item["reference"]
            ][:10],
            "batch_findings": batch_findings,
        },
        "best_source_id": matched_row.id if matched_row else None,
        "best_source": reference_name or None,
        "sources": [
            {"source_id": row.id, "source": Path(row.file_path).name, "title": row.title, "retrieval_score": round(score, 4)}
            for score, row in ranked if reference_name and (Path(row.file_path).name == reference_name or row.title == reference_name)
        ],
        "tfidf_score": None,
        "semantic_score": None,
        "hybrid_score": None,
        "novelty_score": None,
    }

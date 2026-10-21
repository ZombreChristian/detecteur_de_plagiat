"""Moteur RAG autonome dédié à la vérification des TDR.

La recherche lexicale et sémantique classe toutes les références indexées.
Qwen analyse ensuite les cinq meilleures candidates et renvoie un verdict global unique.
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
# uniquement sur les cinq références les mieux classées.
RAG_TOP_K = 5
REFERENCE_CHARS = 3800
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


def _load_all_indexed_references():
    import django
    django.setup()
    from django.db.models import Q
    from detector.models import StudyDocument
    rows = list(
        StudyDocument.objects.filter(
            is_reference=True, document_type__in=["TDR", "RAPPORT"]
        ).filter(Q(cleaned_text__gt="") | Q(extracted_text__gt=""))
        .exclude(embedding=[])
        .order_by("id")
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
    rows = _load_all_indexed_references()
    if not rows:
        raise IndependentTDRError(
            "Aucun TDR ni rapport d'étude de référence n'est présent dans l'index PostgreSQL. "
            "Préparez l'index avant de lancer l'analyse."
        )

    # Le classement sert uniquement à retrouver les références candidates.
    # Qwen reste responsable du raisonnement et du verdict métier. On lui fournit
    # cinq références pour limiter les faux négatifs dus à une reformulation.
    ranked = _rank_all_references(candidate, rows)
    if not ranked:
        raise IndependentTDRError("Aucune référence TDR exploitable n'a été trouvée.")
    selected = ranked[:RAG_TOP_K]
    candidate_excerpt = _excerpt(candidate, CANDIDATE_CHARS)

    reference_blocks = []
    for rank, (hybrid, lexical, semantic, row) in enumerate(selected, start=1):
        reference_blocks.append(
            "DOCUMENT DE RÉFÉRENCE\n"
            f"TITRE: {row.title}\nTYPE DE DOCUMENT: {row.get_document_type_display()}\nFICHIER: {Path(row.file_path).name}\n"
            f"SCORES DE RÉCUPÉRATION (indicatifs seulement): lexical={lexical:.3f}, "
            f"sémantique={semantic:.3f}, combiné={hybrid:.3f}\n"
            f"CONTENU DU DOCUMENT DE RÉFÉRENCE:\n"
            f"{_excerpt(row.extracted_text or row.cleaned_text, REFERENCE_CHARS)}"
        )

    system_prompt = (
        "Tu es l'analyste principal chargé d'évaluer si le TDR candidat reprend, "
        "reformule ou recoupe une étude déjà présente dans la base. Le raisonnement "
        "de Qwen est la base du verdict final : les scores de recherche servent à "
        "retrouver les références, mais ne décident jamais à ta place. Tu disposes "
        "de cinq références candidates. Examine leur contenu et compare le sens, pas "
        "seulement les mots exacts : objet, problème traité, objectif général et "
        "objectifs spécifiques, activités, zone, population cible, période, méthode "
        "et livrables. Une reformulation, des synonymes ou un titre différent ne "
        "suffisent PAS à conclure qu'il s'agit d'une nouvelle étude.\n"
        "GRILLE DE DÉCISION OBLIGATOIRE : "
        "CAS_1_CORRESPONDANCE_FORTE = même étude ou quasi-identique, notamment si "
        "l'objet et les objectifs essentiels coïncident malgré une reformulation ; "
        "CAS_2_CORRESPONDANCE_PARTIELLE = correspondance faible à moyenne, plusieurs "
        "éléments importants communs mais recouvrement incomplet ou différences "
        "substantielles ; CAS_3_MEME_OBJECTIF_ZONE_DIFFERENTE = objectif et démarche "
        "comparables mais zone géographique réellement différente ; "
        "CAS_4_AUCUNE_ETUDE_PERTINENTE = seulement des ressemblances générales ou "
        "aucune référence suffisamment liée au sujet.\n"
        "IMPORTANT : une correspondance faible mais réelle doit être signalée en "
        "CAS_2, et non masquée par CAS_4. CAS_4 n'est pas un choix par défaut lorsque "
        "les documents sont reformulés ou lorsque les preuves sont imparfaites. "
        "À l'inverse, ne déclare pas une correspondance sur la seule présence de mots "
        "génériques. Explique précisément les indices qui soutiennent le niveau choisi "
        "et distingue les faits observés des incertitudes. Les scores lexical, "
        "sémantique et combiné sont des indices de récupération, pas des probabilités "
        "ni des verdicts. Ne prétends jamais qu'une étude a été réalisée à partir d'un TDR seul. "
        "Un TDR prouve qu'un TDR est enregistré ; un rapport d'étude indique qu'un rapport "
        "correspondant est présent dans le registre. N'affirme rien au-delà du contenu réel "
        "du document.\n"
        "COHÉRENCE OBLIGATOIRE : le case_id, le decision, le correspondence_level, "
        "l'explication et la recommandation doivent tous exprimer la même conclusion. "
        "Si tu écris que le candidat correspond fortement à un document, que les objectifs, "
        "la zone, la méthode et les livrables sont identiques ou très proches, le verdict "
        "ne peut pas être CAS_4. Choisis CAS_1 si c'est la même étude ou quasi-identique, "
        "CAS_2 si la correspondance est réelle mais partielle. CAS_4 exige une explication "
        "qui constate l'absence de référence pertinente. Avant de répondre, vérifie cette "
        "cohérence et corrige le JSON si nécessaire.\n"
        "Les seules références que tu peux citer sont celles listées ci-dessous, avec leur "
        "identifiant, leur titre et leur nom de fichier exacts. N'invente jamais un identifiant "
        "comme TDR-TEST-101 s'il ne figure pas dans la liste. Pour matched_reference, recopie "
        "exactement le titre ou le nom de fichier d'une référence fournie ; si aucune référence "
        "ne convient, renvoie une chaîne vide.\n"
        "Choisis un seul verdict global et donne les points communs, les différences et une "
        "recommandation. Les libellés internes de récupération ne sont pas destinés à "
        "l'utilisateur : ne dis jamais « TDR 1 », « TDR 2 », « candidat de recherche 1/2 », "
        "ni « document numéro 1/2 ». N'invente aucun fait. Réponds uniquement avec un objet "
        "JSON valide ayant les clés case_id, decision, explanation, matched_reference, confidence, "
        "common_points (tableau), differences (tableau), recommendation, correspondence_level "
        "(forte, moyenne, faible ou nulle), evidence_from_reference (tableau de courts éléments "
        "concrets).\n"
        "Rappel : les références sont des TDR ou des rapports d'étude présents dans la base ; ne prétends pas "
        "qu'une étude a été exécutée si le document ne l'établit pas."
    )
    final = _ollama_json(
        system_prompt,
        "TDR CANDIDAT À ÉVALUER:\n" + candidate_excerpt
        + "\n\nLES CINQ RÉFÉRENCES À EXAMINER AVANT DE CONCLURE:\n"
        + "\n\n--- DOCUMENT SUIVANT ---\n\n".join(reference_blocks)
        + "\n\nLISTE AUTORISÉE DES RÉFÉRENCES (reprends exactement un titre ou fichier de cette liste) :\n"
        + "\n".join(
            f"- ID={row.id} | TYPE={row.get_document_type_display()} | TITRE={row.title} | FICHIER={Path(row.file_path).name}"
            for _, _, _, row in selected
        ),
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
        "CAS_1_CORRESPONDANCE_FORTE": "Correspondance forte — document correspondant trouvé dans le registre",
        "CAS_2_CORRESPONDANCE_PARTIELLE": "Correspondance partielle — vérification recommandée",
        "CAS_3_MEME_OBJECTIF_ZONE_DIFFERENTE": "Objectifs similaires — zone géographique différente",
        "CAS_4_AUCUNE_ETUDE_PERTINENTE": "Aucun document de référence pertinent trouvé dans la base",
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
            "correspondence_level": str(final.get("correspondence_level", "À confirmer"))[:40],
            "evidence_from_reference": [
                str(x)[:500] for x in final.get("evidence_from_reference", [])[:8]
            ] if isinstance(final.get("evidence_from_reference", []), list) else [],
            "references_examined": len(selected),
            "retrieval_candidates": [
                {
                    "rank": rank,
                    "source_id": row.id,
                    "title": str(row.title or ""),
                    "filename": Path(row.file_path).name,
                    "document_type": row.get_document_type_display(),
                    "lexical_score": round(float(lexical), 4),
                    "semantic_score": round(float(semantic), 4),
                    "hybrid_score": round(float(hybrid), 4),
                }
                for rank, (hybrid, lexical, semantic, row) in enumerate(selected, start=1)
            ],
            "corpus_references_ranked": len(ranked),
            "batches_examined": 1,
            "evidence": evidence,
            # Les références restent des éléments de comparaison, pas des verdicts séparés : un seul verdict global.
            "batch_findings": [],
        },
        "best_source_id": matched_row.id if matched_row else None,
        "best_source": reference_name or None,
        "sources": [
            {
                "source_id": matched_row.id,
                "source": Path(matched_row.file_path).name,
                "title": matched_row.title,
                "document_type": matched_row.get_document_type_display(),
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

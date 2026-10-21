"""Moteur RAG autonome dédié à la vérification des TDR.

La recherche lexicale et sémantique classe toutes les références indexées.
Qwen analyse ensuite les trois meilleures candidates et renvoie un verdict global unique.
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
OLLAMA_MODEL = os.getenv("TDR_RAG_MODEL", os.getenv("OLLAMA_MODEL", "qwen3:14b"))
OLLAMA_TIMEOUT = max(60, int(os.getenv("TDR_RAG_TIMEOUT_SECONDS", "300")))
# Limiter le contexte et le lot d’évaluation aide les machines à mémoire partagée.
TDR_RAG_NUM_CTX = max(4096, int(os.getenv("TDR_RAG_NUM_CTX", "4096")))
TDR_RAG_NUM_BATCH = min(256, max(32, int(os.getenv("TDR_RAG_NUM_BATCH", "64"))))
# Recherche lexicale + sémantique sur le corpus, puis arbitrage sur les trois références les mieux classées.
RAG_TOP_K = min(3, max(1, int(os.getenv("TDR_RAG_TOP_K", "3"))))
REFERENCE_CHARS = min(2400, max(1200, int(os.getenv("TDR_RAG_REFERENCE_CHARS", "1700"))))
CANDIDATE_CHARS = min(6000, max(4000, int(os.getenv("TDR_RAG_CANDIDATE_CHARS", "5000"))))
OUTPUT_TOKENS = min(900, max(500, int(os.getenv("TDR_RAG_INDEPENDENT_MAX_OUTPUT_TOKENS", "700"))))

CASES = {
    "CAS_1_CORRESPONDANCE_FORTE",
    "CAS_2_CORRESPONDANCE_PARTIELLE",
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
                    "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
                    "options": {
                        "temperature": 0,
                        "num_predict": token_budget,
                        "num_ctx": TDR_RAG_NUM_CTX,
                        "num_batch": TDR_RAG_NUM_BATCH,
                    },
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
    # Calcul vectorisé des cosinus : évite une boucle Python sur chaque référence.
    embeddings = np.asarray([row.embedding for row in rows], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    semantic_scores = np.divide(
        embeddings @ candidate_vector,
        norms,
        out=np.zeros(len(rows), dtype=np.float32),
        where=norms > 0,
    )
    semantic_scores = np.clip(semantic_scores, 0.0, 1.0)
    lexical_scores = np.clip(np.asarray(lexical_scores, dtype=np.float32), 0.0, 1.0)
    hybrid_scores = 0.30 * lexical_scores + 0.70 * semantic_scores
    ranked = [
        (float(hybrid_scores[i]), float(lexical_scores[i]), float(semantic_scores[i]), row)
        for i, row in enumerate(rows)
    ]
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
    # Qwen reste responsable du raisonnement et du verdict métier ; seules les
    # trois références les mieux classées lui sont transmises.
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
        "des trois références candidates les mieux classées. Examine leur contenu et compare le sens, pas "
        "seulement les mots exacts : objet, problème traité, objectif général et "
        "objectifs spécifiques, activités, zone, population cible, période, méthode "
        "et livrables. Une reformulation, des synonymes ou un titre différent ne "
        "suffisent PAS à conclure qu'il s'agit d'une nouvelle étude.\n"
        "GRILLE DE DÉCISION OBLIGATOIRE : "
        "CAS_1_CORRESPONDANCE_FORTE = même étude ou quasi-identique, notamment si "
        "l'objet et les objectifs essentiels coïncident malgré une reformulation ; "
        "CAS_2_CORRESPONDANCE_PARTIELLE = une ou plusieurs références partagent des "
        "éléments métier significatifs, même si la zone, la méthode ou les livrables "
        "diffèrent ; CAS_4_AUCUNE_ETUDE_PERTINENTE = aucune référence ne présente "
        "de correspondance métier significative. Une zone différente peut relever "
        "de CAS_2 si les objectifs et la démarche se recoupent réellement.\n"
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
        "Évalue séparément CHAQUE référence. Retourne matched_references, un tableau "
        "de toutes les références réellement fortes ou partielles, chacune avec filename, "
        "level (forte ou partielle), explanation, common_points et differences. Le verdict "
        "global est CAS_1 si au moins une référence est forte ; CAS_2 si aucune n'est forte "
        "mais qu'au moins une est partielle ; CAS_4 si aucune n'est pertinente. "
        "Choisis aussi un seul verdict global et donne les points communs, les différences et "
        "une recommandation. Ne cite que les noms exacts de la liste autorisée. Les libellés "
        "internes de récupération ne sont pas destinés à l'utilisateur : ne dis jamais "
        "« TDR 1 », « TDR 2 » ou « document numéro 1/2 ». N'invente aucun fait. Réponds "
        "uniquement avec un objet JSON valide ayant les clés case_id, decision, explanation, "
        "matched_reference, matched_references, confidence, common_points (tableau), differences "
        "(tableau), recommendation, correspondence_level (forte, moyenne ou nulle), "
        "evidence_from_reference (tableau).\n"
        "Rappel : les références sont des TDR ou des rapports d'étude présents dans la base ; ne prétends pas "
        "qu'une étude a été exécutée si le document ne l'établit pas."
    )
    final = _ollama_json(
        system_prompt,
        "TDR CANDIDAT À ÉVALUER:\n" + candidate_excerpt
        + "\n\nLES TROIS RÉFÉRENCES À EXAMINER AVANT DE CONCLURE:\n"
        + "\n\n--- DOCUMENT SUIVANT ---\n\n".join(reference_blocks)
        + "\n\nLISTE AUTORISÉE DES RÉFÉRENCES (reprends exactement un titre ou fichier de cette liste) :\n"
        + "\n".join(
            f"- ID={row.id} | TYPE={row.get_document_type_display()} | TITRE={row.title} | FICHIER={Path(row.file_path).name}"
            for _, _, _, row in selected
        ),
        tokens=OUTPUT_TOKENS,
    )

    case_id = str(final.get("case_id", "CAS_4_AUCUNE_ETUDE_PERTINENTE"))
    if case_id not in CASES:
        case_id = "CAS_4_AUCUNE_ETUDE_PERTINENTE"

    # Réconcilie les contradictions connues du modèle : si son texte affirme
    # explicitement une forte correspondance mais que son case_id vaut CAS_4,
    # ne laisse pas l'interface afficher deux conclusions opposées.
    explanation_text = str(final.get("explanation", ""))
    recommendation_text = str(final.get("recommendation", ""))
    common_points = final.get("common_points", [])
    differences = final.get("differences", [])
    reasoning_text = " ".join([
        explanation_text,
        recommendation_text,
        " ".join(str(x) for x in common_points) if isinstance(common_points, list) else "",
    ]).casefold()
    strong_phrases = (
        "correspond fortement", "correspondance forte", "identiques ou très proches",
        "même étude", "meme etude", "reformulation très proche",
        "reformulation tres proche", "ne pas enregistrer", "même tdr",
        "meme tdr", "contenu et la structure", "alignés avec",
    )
    partial_phrases = (
        "correspondance partielle", "proche mais", "objectifs comparables",
        "similitude méthodologique", "similitude methodologique",
    )
    if case_id == "CAS_4_AUCUNE_ETUDE_PERTINENTE":
        if any(phrase in reasoning_text for phrase in strong_phrases):
            case_id = "CAS_1_CORRESPONDANCE_FORTE"
        elif any(phrase in reasoning_text for phrase in partial_phrases):
            case_id = "CAS_2_CORRESPONDANCE_PARTIELLE"

    # Ne conserver que les références effectivement transmises au modèle.
    allowed_rows = {Path(row.file_path).name: row for _, _, _, row in selected}
    allowed_rows.update({str(row.title or ""): row for _, _, _, row in selected})
    model_refs = final.get("matched_references", [])
    if not isinstance(model_refs, list):
        model_refs = []
    requested = []
    for item in model_refs:
        name = str(item.get("filename") or item.get("source") or item.get("title") or "").strip() if isinstance(item, dict) else str(item).strip()
        if name and name not in requested:
            requested.append(name)
    legacy = str(final.get("matched_reference", "")).strip()
    if legacy and legacy not in requested:
        requested.insert(0, legacy)
    matched_rows = []
    for name in requested:
        row = allowed_rows.get(name)
        if row is None:
            row = next((candidate for _, _, _, candidate in selected
                        if name.casefold() in Path(candidate.file_path).name.casefold()
                        or name.casefold() in str(candidate.title or "").casefold()), None)
        if row is not None and row not in matched_rows:
            matched_rows.append(row)
    strong_names = {
        str(item.get("filename") or item.get("source") or item.get("title") or "").strip()
        for item in model_refs if isinstance(item, dict) and str(item.get("level", "")).casefold() == "forte"
    }
    if matched_rows and any(name in strong_names for name in requested):
        case_id = "CAS_1_CORRESPONDANCE_FORTE"
    elif matched_rows:
        case_id = "CAS_2_CORRESPONDANCE_PARTIELLE"
    if case_id == "CAS_4_AUCUNE_ETUDE_PERTINENTE":
        matched_rows = []
    if not matched_rows and case_id != "CAS_4_AUCUNE_ETUDE_PERTINENTE":
        matched_rows = [selected[0][3]]
    matched_row = matched_rows[0] if matched_rows else None
    reference_name = Path(matched_row.file_path).name if matched_row else ""
    case_titles = {
        "CAS_1_CORRESPONDANCE_FORTE": "Forte correspondance",
        "CAS_2_CORRESPONDANCE_PARTIELLE": "Correspondance partielle",
        "CAS_4_AUCUNE_ETUDE_PERTINENTE": "Aucune correspondance",
    }
    per_reference = []
    for row in matched_rows:
        item = next((item for item in model_refs if isinstance(item, dict)
                     and str(item.get("filename") or item.get("source") or item.get("title") or "").strip()
                     in {Path(row.file_path).name, str(row.title or "")}), {})
        level = str(item.get("level", "")).casefold()
        per_reference.append({
            "source_id": row.id,
            "source": Path(row.file_path).name,
            "title": str(row.title or ""),
            "document_type": row.get_document_type_display(),
            "level": "forte" if level == "forte" or (not level and case_id == "CAS_1_CORRESPONDANCE_FORTE" and row is matched_rows[0]) else "partielle",
            "explanation": str(item.get("explanation") or final.get("explanation", ""))[:1200],
            "common_points": [str(x)[:400] for x in item.get("common_points", final.get("common_points", []))[:8]],
            "differences": [str(x)[:400] for x in item.get("differences", final.get("differences", []))[:8]],
        })
    evidence = [{
        "source": item["source"],
        "point": item["explanation"][:900],
        "justification": "Référence retenue après examen du contenu par le RAG.",
    } for item in per_reference]
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
            "matched_references": per_reference,
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
            {"source_id": item["source_id"], "source": item["source"],
             "title": item["title"], "document_type": item["document_type"],
             "level": item["level"]}
            for item in per_reference
        ],
        "tfidf_score": None,
        "semantic_score": None,
        "hybrid_score": None,
        "novelty_score": None,
    }

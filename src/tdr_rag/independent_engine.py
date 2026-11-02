"""Moteur RAG autonome dédié à la vérification des TDR.

La recherche lexicale et sémantique classe toutes les références indexées.
Le LLM analyse ensuite les trois meilleures candidates et renvoie un verdict global unique.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend"
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "frontend.settings")

# ---------------------------------------------------------------------------
# Configuration LLM
# ---------------------------------------------------------------------------
# Le fournisseur distant compatible OpenAI est utilisé par défaut pour l'arbitrage TDR.
# Ollama reste disponible si TDR_LLM_PROVIDER=ollama est explicitement configuré.
TDR_LLM_PROVIDER = os.getenv("TDR_LLM_PROVIDER", "openai_compatible").strip().lower()
TDR_LLM_BASE_URL = (
    os.getenv("TDR_LLM_BASE_URL")
    or "https://meyer-variable-halo-labeled.trycloudflare.com/api"
).strip().rstrip("/")
TDR_LLM_API_KEY = os.getenv("TDR_LLM_API_KEY", "").strip()
TDR_LLM_REMOTE_MODEL = (os.getenv("TDR_LLM_MODEL") or "qwen3.6-35b").strip()
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("TDR_RAG_MODEL", os.getenv("OLLAMA_MODEL", "qwen3:8b"))
REMOTE_PROVIDERS = {"openai_compatible", "openai-compatible", "remote"}
ACTIVE_MODEL = (
    TDR_LLM_REMOTE_MODEL
    if TDR_LLM_PROVIDER in REMOTE_PROVIDERS and TDR_LLM_REMOTE_MODEL
    else OLLAMA_MODEL
)
OLLAMA_TIMEOUT = max(60, int(os.getenv("TDR_RAG_TIMEOUT_SECONDS", "300")))

# Recherche rapide lexical + sémantique sur l'ensemble du corpus, puis arbitrage RAG
# uniquement sur les trois références les mieux classées.
RAG_TOP_K = 3
REFERENCE_CHARS = 2200
CANDIDATE_CHARS = 6000
# Les serveurs Qwen en mode raisonnement peuvent consommer tout le budget
# avant de produire le JSON final. Les tentatives augmentent donc le budget
# et, sur les reprises, désactivent explicitement le thinking.
OUTPUT_TOKENS = 2048

# ---------------------------------------------------------------------------
# Identifiants de cas : UNE SEULE source de vérité (prompt + code)
# ---------------------------------------------------------------------------
CASE_1 = "CAS_1_correspondance forte à une étude existante"
CASE_2 = "CAS_2_Correspondance partielle avec une étude existante"
CASE_3 = "CAS_3_Aucune correspondance à une étude existante"

CASES = {CASE_1, CASE_2, CASE_3}

CASE_TITLES = {
    CASE_1: "Forte correspondance",
    CASE_2: "Correspondance partielle",
    CASE_3: "Aucune correspondance",
}

SYSTEM_PROMPT = f"""Tu es l'analyste chargé de déterminer si un TDR candidat reprend, reformule ou recoupe un document déjà présent dans la base (TDR ou rapport d'étude).

# MÉTHODE
1. Lis le TDR candidat, puis les trois références fournies.
2. Compare le SENS, pas les mots : objet, problème traité, objectif général et spécifiques, activités, zone, population cible, période, méthode, livrables. Une reformulation, des synonymes ou un titre différent ne suffisent pas à conclure à une étude nouvelle.
3. Compare les trois références en interne, puis retiens UNE SEULE référence : celle dont l'objet et les objectifs sont les plus proches (en cas d'égalité, celle aux preuves les plus solides).
4. Ne mentionne jamais les références écartées, ni « TDR 1 / TDR 2 », ni les scores de recherche (ce sont des indices de récupération, pas des preuves).

# GRILLE DE DÉCISION (case_id : recopie la chaîne exacte)
- "{CASE_1}" : même étude ou quasi identique (objet et objectifs essentiels coïncident, malgré une reformulation).
- "{CASE_2}" : éléments métier significatifs en commun (objectifs, démarche, activités), même si la zone, la méthode ou les livrables diffèrent. Une zone différente reste un CAS_2 si objectifs et démarche se recoupent. Une correspondance faible mais réelle relève de ce cas.
- "{CASE_3}" : aucune des trois références ne présente de correspondance métier significative.
Ne choisis pas CAS_3 par défaut quand les preuves sont imparfaites. Ne déclare pas de correspondance sur de simples mots génériques (« évaluation », « développement », « diagnostic »).

# RÈGLES DE FOND
- Appuie-toi uniquement sur le contenu fourni. Distingue les faits observés des incertitudes.
- Un TDR prouve seulement qu'un TDR est enregistré ; un rapport prouve seulement qu'un rapport est enregistré. N'affirme jamais qu'une étude a été réalisée si le document ne l'établit pas.
- Ne cite que des références de la LISTE AUTORISÉE, avec leur nom de fichier exact. N'invente aucun identifiant.

# COHÉRENCE (vérifie avant de répondre)
case_id, decision, correspondence_level, explanation et recommendation doivent exprimer la même conclusion.
- CAS_1 → correspondence_level "forte", matched_references[0].level "forte"
- CAS_2 → correspondence_level "moyenne", matched_references[0].level "partielle"
- CAS_3 → correspondence_level "nulle", matched_reference "", matched_references [], evidence_from_reference []
Si ton texte décrit une forte correspondance, le case_id ne peut pas être CAS_3. Corrige le JSON si besoin.

# SORTIE
Réponds uniquement par un objet JSON valide (sans Markdown, sans texte autour) :
{{
  "case_id": "<une des 3 chaînes exactes>",
  "decision": "<phrase courte résumant le verdict>",
  "correspondence_level": "forte | moyenne | nulle",
  "confidence": "élevée | moyenne | faible",
  "matched_reference": "<nom de fichier exact, ou ''>",
  "matched_references": [
    {{
      "filename": "<nom de fichier exact>",
      "level": "forte | partielle",
      "explanation": "<justification concrète, 3 à 5 phrases>",
      "common_points": ["..."],
      "differences": ["..."]
    }}
  ],
  "explanation": "<justification globale fondée sur des éléments observés>",
  "common_points": ["..."],
  "differences": ["..."],
  "evidence_from_reference": ["<passage ou fait précis de la référence retenue>"],
  "recommendation": "<1 à 2 phrases, opérationnelles>"
}}
Sois concis : termine toujours le JSON."""


class IndependentTDRError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Appel LLM
# ---------------------------------------------------------------------------
def _validate_llm_config() -> None:
    """Erreurs de configuration : inutile de les retenter."""
    if TDR_LLM_PROVIDER in REMOTE_PROVIDERS:
        if not TDR_LLM_BASE_URL:
            raise IndependentTDRError(
                "TDR_LLM_BASE_URL est requis pour le fournisseur openai_compatible."
            )
        if not TDR_LLM_API_KEY:
            raise IndependentTDRError(
                "TDR_LLM_API_KEY est requis pour le fournisseur openai_compatible."
            )
        if not TDR_LLM_REMOTE_MODEL:
            raise IndependentTDRError(
                "TDR_LLM_MODEL est requis pour le fournisseur openai_compatible."
            )
    elif TDR_LLM_PROVIDER != "ollama":
        raise IndependentTDRError(
            "TDR_LLM_PROVIDER invalide. Valeurs acceptées : ollama, openai_compatible."
        )


def _ollama_json(system_prompt: str, user_prompt: str, tokens: int = OUTPUT_TOKENS) -> dict:
    """Appelle le fournisseur configuré avec reprises limitées en cas de JSON invalide."""
    _validate_llm_config()

    last_error = None
    last_content = ""
    llm_audit = {
        "provider": TDR_LLM_PROVIDER,
        "requested_model": ACTIVE_MODEL,
        "reported_model": None,
        "request_id": None,
    }
    token_budgets = (max(tokens, 2048), max(tokens, 4096), max(tokens, 6144))

    for attempt in range(3):
        token_budget = token_budgets[attempt]
        retry_note = ""
        if attempt:
            retry_note = (
                "\n\nIMPORTANT — la réponse précédente était invalide ou incomplète. "
                "Recommence et renvoie un unique objet JSON valide, complet, sans Markdown, "
                "sans texte avant/après, avec toutes les chaînes correctement échappées. "
                "Reste concis et termine toutes les accolades et tous les guillemets."
            )
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + retry_note},
            ]
            if TDR_LLM_PROVIDER in REMOTE_PROVIDERS:
                from openai import OpenAI

                client = OpenAI(
                    base_url=TDR_LLM_BASE_URL,
                    api_key=TDR_LLM_API_KEY,
                    timeout=OLLAMA_TIMEOUT,
                    max_retries=0,
                )
                request_kwargs = {
                    "model": TDR_LLM_REMOTE_MODEL,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": token_budget,
                }
                if attempt == 0:
                    # Premier essai : mode JSON strict.
                    request_kwargs["response_format"] = {"type": "json_object"}
                else:
                    # Reprises : certains serveurs gèrent mal response_format ; on
                    # désactive le thinking (vLLM + template Qwen).
                    request_kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                response = client.chat.completions.create(**request_kwargs)
                llm_audit["reported_model"] = getattr(response, "model", None)
                llm_audit["request_id"] = getattr(response, "id", None)
                if not response.choices:
                    raise ValueError("Le serveur LLM n'a renvoyé aucun choix (choices vide).")
                choice = response.choices[0]
                message = choice.message
                last_content = (getattr(message, "content", None) or "").strip()
                finish_reason = getattr(choice, "finish_reason", None)
                if not last_content:
                    refusal = (getattr(message, "refusal", None) or "").strip()
                    reasoning = (getattr(message, "reasoning_content", None) or "").strip()
                    details = []
                    if finish_reason:
                        details.append(f"finish_reason={finish_reason}")
                    if refusal:
                        details.append(f"refusal={refusal[:160]}")
                    if reasoning:
                        details.append("reasoning_content présent mais content vide")
                    if finish_reason == "length":
                        details.append("budget de génération épuisé avant la réponse finale")
                    raise ValueError(
                        "Réponse vide du serveur OpenAI-compatible"
                        + (f" ({'; '.join(details)})" if details else "")
                        + ". Nouvelle tentative avec budget accru et thinking désactivé."
                    )
            else:  # ollama
                response = requests.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": ACTIVE_MODEL,
                        "stream": False,
                        "format": "json",
                        "think": False,
                        "keep_alive": "5m",
                        "options": {"temperature": 0, "num_predict": token_budget},
                        "messages": messages,
                    },
                    timeout=OLLAMA_TIMEOUT,
                )
                response.raise_for_status()
                payload = response.json()
                llm_audit["reported_model"] = payload.get("model") or OLLAMA_MODEL
                last_content = (payload.get("message") or {}).get("content", "").strip()

            if not last_content:
                raise ValueError("Le fournisseur LLM a renvoyé une réponse vide.")
            parsed = json.loads(last_content)
            if not isinstance(parsed, dict):
                raise ValueError("La réponse JSON n'est pas un objet.")
            parsed["_llm_audit"] = dict(llm_audit)
            return parsed
        except Exception as exc:
            last_error = exc
            logger.warning(
                "TDR LLM tentative %d/3 échouée (modèle=%s, budget=%d) : %s",
                attempt + 1, ACTIVE_MODEL, token_budget, exc,
            )
            # Cloudflare 524 : le proxy a expiré après ~120 s. Ne pas relancer
            # immédiatement une génération qui vient de dépasser la limite.
            if getattr(exc, "status_code", None) == 524 or "Error code: 524" in str(exc):
                raise IndependentTDRError(
                    "Le serveur LLM distant a dépassé le délai Cloudflare (HTTP 524, "
                    "environ 120 secondes). La génération a été interrompue ; aucun verdict "
                    "incomplet n'a été enregistré. Attendez au moins 120 secondes puis "
                    "relancez l'analyse. Si l'erreur persiste, le serveur distant doit être "
                    "vérifié par son administrateur."
                ) from exc

    preview = re.sub(r"\s+", " ", last_content)[:240]
    raise IndependentTDRError(
        f"Le fournisseur LLM ({ACTIVE_MODEL}) n'a pas produit de JSON valide après 3 tentatives. "
        f"Dernière erreur : {last_error}. Début de la réponse : {preview!r}"
    ) from last_error


# ---------------------------------------------------------------------------
# Extraction et préparation des textes
# ---------------------------------------------------------------------------
def extract_tdr_document(path: Path) -> str:
    """Extraction autonome du TDR candidat, sans passer par l'ancien moteur."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = ((page.extract_text() or "").strip() for page in reader.pages)
        return "\n\n".join(text for text in pages if text)
    if suffix == ".docx":
        from docx import Document

        document = Document(str(path))
        parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
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


# ---------------------------------------------------------------------------
# Récupération / classement
# ---------------------------------------------------------------------------
def _load_all_indexed_references():
    import django

    django.setup()
    from django.db.models import Q
    from detector.models import StudyDocument

    return list(
        StudyDocument.objects.filter(
            is_reference=True, document_type__in=["TDR", "RAPPORT"]
        ).filter(Q(cleaned_text__gt="") | Q(extracted_text__gt=""))
        .exclude(embedding=[])
        .order_by("id")
    )


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


# ---------------------------------------------------------------------------
# Utilitaires de post-traitement
# ---------------------------------------------------------------------------
def _as_list(value, limit: int, max_chars: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x)[:max_chars] for x in value[:limit]]


def _item_name(item) -> str:
    if isinstance(item, dict):
        return str(item.get("filename") or item.get("source") or item.get("title") or "").strip()
    return str(item).strip()


STRONG_PHRASES = (
    "correspond fortement", "correspondance forte", "identiques ou très proches",
    "même étude", "meme etude", "reformulation très proche",
    "reformulation tres proche", "ne pas enregistrer", "même tdr",
    "meme tdr", "contenu et la structure", "alignés avec",
)
PARTIAL_PHRASES = (
    "correspondance partielle", "proche mais", "objectifs comparables",
    "similitude méthodologique", "similitude methodologique",
)


# ---------------------------------------------------------------------------
# Analyse principale
# ---------------------------------------------------------------------------
def analyze_tdr_independently(candidate_text: str) -> dict:
    """Classe le corpus, puis demande au LLM un seul verdict fondé sur les 3 meilleurs TDR."""
    candidate = (candidate_text or "").strip()
    if len(candidate) < 80:
        raise IndependentTDRError("Le TDR contient trop peu de texte exploitable.")
    rows = _load_all_indexed_references()
    if not rows:
        raise IndependentTDRError(
            "Aucun TDR ni rapport d'étude de référence n'est présent dans l'index PostgreSQL. "
            "Préparez l'index avant de lancer l'analyse."
        )

    # Le classement sert uniquement à retrouver les références candidates ;
    # le LLM reste responsable du verdict métier.
    ranked = _rank_all_references(candidate, rows)
    if not ranked:
        raise IndependentTDRError("Aucune référence TDR exploitable n'a été trouvée.")
    selected = ranked[:RAG_TOP_K]
    candidate_excerpt = _excerpt(candidate, CANDIDATE_CHARS)

    reference_blocks = []
    for _, _, _, row in selected:
        reference_blocks.append(
            "DOCUMENT DE RÉFÉRENCE\n"
            f"ID: {row.id}\n"
            f"TITRE: {row.title}\n"
            f"TYPE DE DOCUMENT: {row.get_document_type_display()}\n"
            f"FICHIER: {Path(row.file_path).name}\n"
            f"CONTENU:\n{_excerpt(row.extracted_text or row.cleaned_text, REFERENCE_CHARS)}"
        )

    user_prompt = (
        "TDR CANDIDAT À ÉVALUER:\n" + candidate_excerpt
        + "\n\nLES TROIS RÉFÉRENCES LES MIEUX CLASSÉES À EXAMINER AVANT DE CONCLURE:\n"
        + "\n\n--- DOCUMENT SUIVANT ---\n\n".join(reference_blocks)
        + "\n\nLISTE AUTORISÉE DES RÉFÉRENCES (reprends exactement un nom de fichier de cette liste) :\n"
        + "\n".join(
            f"- ID={row.id} | TYPE={row.get_document_type_display()} | "
            f"TITRE={row.title} | FICHIER={Path(row.file_path).name}"
            for _, _, _, row in selected
        )
    )
    final = _ollama_json(SYSTEM_PROMPT, user_prompt, tokens=OUTPUT_TOKENS)

    llm_audit = final.pop("_llm_audit", {
        "provider": TDR_LLM_PROVIDER,
        "requested_model": ACTIVE_MODEL,
        "reported_model": None,
        "request_id": None,
    })

    case_id = str(final.get("case_id", CASE_3))
    if case_id not in CASES:
        logger.warning("case_id inconnu renvoyé par le modèle : %r -> repli sur CAS_3", case_id)
        case_id = CASE_3

    # --- Garde-fou : contradiction texte/verdict (journalisée pour mesurer la fiabilité) ---
    explanation_text = str(final.get("explanation", ""))
    recommendation_text = str(final.get("recommendation", ""))
    common_points_raw = final.get("common_points", [])
    reasoning_text = " ".join([
        explanation_text,
        recommendation_text,
        " ".join(str(x) for x in common_points_raw) if isinstance(common_points_raw, list) else "",
    ]).casefold()
    if case_id == CASE_3:
        if any(p in reasoning_text for p in STRONG_PHRASES):
            logger.warning("Garde-fou : CAS_3 contredit par le texte -> CAS_1")
            case_id = CASE_1
        elif any(p in reasoning_text for p in PARTIAL_PHRASES):
            logger.warning("Garde-fou : CAS_3 contredit par le texte -> CAS_2")
            case_id = CASE_2

    # --- Références effectivement transmises au modèle ---
    allowed_rows = {Path(row.file_path).name: row for _, _, _, row in selected}
    allowed_rows.update({str(row.title or ""): row for _, _, _, row in selected})

    model_refs = final.get("matched_references", [])
    if not isinstance(model_refs, list):
        model_refs = []

    requested: list[str] = []
    for item in model_refs:
        name = _item_name(item)
        if name and name not in requested:
            requested.append(name)
    legacy = str(final.get("matched_reference", "")).strip()
    if legacy and legacy not in requested:
        requested.insert(0, legacy)

    matched_rows = []
    for name in requested:
        row = allowed_rows.get(name)
        if row is None:
            needle = name.casefold()
            row = next(
                (c for _, _, _, c in selected
                 if needle in Path(c.file_path).name.casefold()
                 or needle in str(c.title or "").casefold()),
                None,
            )
        if row is not None and row not in matched_rows:
            matched_rows.append(row)

    def _model_item_for(row) -> dict:
        names = {Path(row.file_path).name, str(row.title or "")}
        return next(
            (i for i in model_refs if isinstance(i, dict) and _item_name(i) in names),
            {},
        )

    def _model_level_for(row) -> str:
        return str(_model_item_for(row).get("level", "")).casefold()

    # Une seule référence présentée : privilégier celle classée « forte » par le modèle.
    if matched_rows:
        matched_rows.sort(key=lambda r: 0 if _model_level_for(r) == "forte" else 1)
        matched_rows = matched_rows[:1]

    # --- Cohérence verdict / référence retenue ---
    if matched_rows:
        if case_id == CASE_3:
            # Le modèle cite une référence : on ne peut pas afficher « aucune correspondance ».
            case_id = CASE_1 if _model_level_for(matched_rows[0]) == "forte" else CASE_2
        elif _model_level_for(matched_rows[0]) == "forte" and case_id == CASE_2:
            case_id = CASE_1
    elif case_id != CASE_3:
        # Verdict CAS_1/CAS_2 sans référence exploitable : on retient la mieux classée.
        matched_rows = [selected[0][3]]

    matched_row = matched_rows[0] if matched_rows else None
    reference_name = Path(matched_row.file_path).name if matched_row else ""

    per_reference = []
    for row in matched_rows:
        item = _model_item_for(row)
        level = str(item.get("level", "")).casefold()
        if level not in {"forte", "partielle"}:
            level = "forte" if case_id == CASE_1 else "partielle"
        per_reference.append({
            "source_id": row.id,
            "source": Path(row.file_path).name,
            "title": str(row.title or ""),
            "document_type": row.get_document_type_display(),
            "level": level,
            "explanation": str(item.get("explanation") or final.get("explanation", ""))[:1200],
            "common_points": _as_list(
                item.get("common_points") or final.get("common_points", []), 8, 400),
            "differences": _as_list(
                item.get("differences") or final.get("differences", []), 8, 400),
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
            "case_title": CASE_TITLES[case_id],
            "model": llm_audit.get("reported_model") or ACTIVE_MODEL,
            "model_audit": llm_audit,
            "decision": str(final.get("decision", case_id)),
            "explanation": str(final.get("explanation", ""))[:2500],
            "matched_reference": reference_name,
            "matched_references": per_reference,
            "confidence": str(final.get("confidence", "À confirmer"))[:80],
            "common_points": _as_list(final.get("common_points", []), 10, 500),
            "differences": _as_list(final.get("differences", []), 10, 500),
            "recommendation": str(final.get("recommendation", ""))[:2000],
            "correspondence_level": str(final.get("correspondence_level", "À confirmer"))[:40],
            "evidence_from_reference": _as_list(
                final.get("evidence_from_reference", []), 8, 500),
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
            # Un seul verdict global : pas de verdicts par référence.
            "batch_findings": [],
        },
        "best_source_id": matched_row.id if matched_row else None,
        "best_source": reference_name or None,
        "sources": [
            {"source_id": i["source_id"], "source": i["source"],
             "title": i["title"], "document_type": i["document_type"],
             "level": i["level"]}
            for i in per_reference
        ],
        "tfidf_score": None,
        "semantic_score": None,
        "hybrid_score": None,
        "novelty_score": None,
    }
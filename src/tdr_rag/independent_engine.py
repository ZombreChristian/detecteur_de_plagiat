"""Moteur RAG autonome dédié à la vérification des TDR.

La recherche lexicale et sémantique classe toutes les références indexées.
Le LLM analyse ensuite les trois meilleures candidates et renvoie un verdict global unique.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Cache mémoire du corpus de recherche : les références sont stables entre deux
# analyses. Les embeddings des segments et la matrice TF-IDF ne sont reconstruits
# que si le contenu du registre change.
_REFERENCE_SEARCH_CACHE = None
_REFERENCE_SEARCH_CACHE_KEY = None
_REFERENCE_SEARCH_CACHE_LOCK = threading.RLock()

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend"
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "frontend.settings")

# ---------------------------------------------------------------------------
# Configuration LLM
# ---------------------------------------------------------------------------
# L'arbitrage TDR utilise exclusivement le serveur distant compatible OpenAI.
TDR_LLM_PROVIDER = "openai_compatible"
TDR_LLM_BASE_URL = (
    os.getenv("TDR_LLM_BASE_URL")
    or "https://meyer-variable-halo-labeled.trycloudflare.com/api"
).strip().rstrip("/")
TDR_LLM_API_KEY = os.getenv("TDR_LLM_API_KEY", "").strip()
TDR_LLM_REMOTE_MODEL = (os.getenv("TDR_LLM_MODEL") or "qwen3.6-35b").strip()
TDR_LLM_TIMEOUT = max(60, int(os.getenv("TDR_LLM_TIMEOUT_SECONDS", "300")))
ACTIVE_MODEL = TDR_LLM_REMOTE_MODEL

# Recherche rapide lexical + sémantique sur l'ensemble du corpus, puis arbitrage RAG
# uniquement sur les trois références les mieux classées.
RAG_TOP_K = 1
# Le LLM reçoit l'intégralité du candidat et de la référence retenue.
# La réduction du contexte vient du choix d'une référence, pas d'une troncature.
REFERENCE_CHARS = None
CANDIDATE_CHARS = None
# Les serveurs Qwen en mode raisonnement peuvent consommer tout le budget
# avant de produire le JSON final. Les tentatives augmentent donc le budget
# et, sur les reprises, désactivent explicitement le thinking.
OUTPUT_TOKENS = 4096

# ---------------------------------------------------------------------------
# Identifiants de cas : UNE SEULE source de vérité (prompt + code)
# ---------------------------------------------------------------------------
CASE_1 = "CAS_1_correspondance forte à une étude existante"
CASE_2 = "CAS_2_Correspondance partielle avec une étude existante"
CASE_3 = "CAS_3_Aucune correspondance à une étude existante"

CASES = {CASE_1, CASE_2, CASE_3}

CASE_TITLES = {
    CASE_1: "Correspondance forte",
    CASE_2: "Correspondance partielle",
    CASE_3: "Aucune correspondance",
}

SYSTEM_PROMPT = f"""Tu es l'analyste chargé de déterminer si un TDR candidat reprend, reformule ou recoupe un TDR déjà présent dans le registre.

# MÉTHODE
1. Lis le TDR candidat, puis les trois références fournies.
2. Compare le SENS, pas les mots : objet, problème traité, objectif général et spécifiques, activités, zone, population cible, période, méthode, livrables. Une reformulation, des synonymes ou un titre différent ne suffisent pas à conclure à une étude nouvelle.
3. Compare les trois références en interne, puis retiens UNE SEULE référence : celle dont l'objet et les objectifs sont les plus proches (en cas d'égalité, celle aux preuves les plus solides).
4. Ne mentionne jamais les références écartées, ni « TDR 1 / TDR 2 », ni les scores de recherche (ce sont des indices de récupération, pas des preuves).

# GRILLE DE DÉCISION (case_id : recopie la chaîne exacte)
- "{CASE_1}" : même étude ou quasi identique (objet et objectifs essentiels coïncident, malgré une reformulation).
- "{CASE_2}" : éléments métier significatifs en commun (objectifs, démarche, activités), même si la zone, la méthode ou les livrables diffèrent. Une zone différente reste un CAS_2 si objectifs et démarche se recoupent. Une correspondance faible mais réelle relève de ce cas.
- "{CASE_3}" : aucune des trois références ne présente de correspondance métier significative.
Si les preuves sont incomplètes ou insuffisantes, n'invente pas de correspondance : demande une vérification. Ne déclare pas de correspondance sur de simples mots génériques (« évaluation », « développement », « diagnostic »).

# RÈGLES DE FOND
- Appuie-toi uniquement sur le contenu fourni. Distingue les faits observés des incertitudes.
- Un TDR prouve seulement qu'un TDR est enregistré ; un rapport prouve seulement qu'un rapport est enregistré. N'affirme jamais qu'une étude a été réalisée si le document ne l'établit pas.
- Ne cite que des références de la LISTE AUTORISÉE, avec leur nom de fichier exact. N'invente aucun identifiant.

# COHÉRENCE (vérifie avant de répondre)
case_id, decision, correspondence_level, explanation et recommendation doivent exprimer la même conclusion.
- CAS_1 → correspondence_level "forte", matched_references[0].level "forte"
- CAS_2 → correspondence_level "moyenne", matched_references[0].level "partielle"
- CAS_3 → correspondence_level "nulle", matched_reference "", matched_references [], evidence_from_reference []
CAS_1/CAS_2 exigent une référence autorisée et des extraits textuels exacts vérifiables dans cette référence. En cas de preuves insuffisantes, retourne CAS_3 avec decision « ANALYSE_INSUFFISANTE ». Le contenu documentaire prime sur la formulation.

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
      "differences": ["..."],
      "source_evidence": ["<au moins deux extraits exacts copiés mot pour mot de cette référence>"],
    }}
  ],
  "explanation": "<justification globale fondée sur des éléments observés>",
  "common_points": ["..."],
  "differences": ["..."],
  "evidence_from_reference": ["<extrait exact copié de la référence retenue>"],
}}
Sois concis : termine toujours le JSON."""


class IndependentTDRError(RuntimeError):
    pass


class InsufficientEvidenceError(IndependentTDRError):
    """Le contenu extrait ne permet pas de confirmer un verdict métier."""
    pass


# ---------------------------------------------------------------------------
# Appel LLM
# ---------------------------------------------------------------------------
def _validate_llm_config() -> None:
    """Vérifie la configuration du fournisseur distant unique."""
    if not TDR_LLM_BASE_URL:
        raise IndependentTDRError("TDR_LLM_BASE_URL est requis.")
    if not TDR_LLM_REMOTE_MODEL:
        raise IndependentTDRError("TDR_LLM_MODEL est requis.")


def _request_llm_json(system_prompt: str, user_prompt: str, tokens: int = OUTPUT_TOKENS) -> dict:
    """Appelle le fournisseur configuré avec reprises limitées en cas de JSON invalide."""
    _validate_llm_config()

    last_error = None
    last_content = ""
    llm_audit = {
        "provider": TDR_LLM_PROVIDER,
        "requested_model": ACTIVE_MODEL,
        "reported_model": None,
        "request_id": None,
        "attempts": 0,
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    token_budgets = (max(tokens, 4096), max(tokens, 6144))

    # Deux appels maximum : la première génération est déjà configurée sans
    # thinking afin d'éviter de payer plusieurs dizaines de secondes de raisonnement
    # avant l'émission du JSON métier.
    for attempt in range(2):
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
            from openai import OpenAI

            client = OpenAI(
                base_url=TDR_LLM_BASE_URL,
                api_key=TDR_LLM_API_KEY or "not-needed",
                timeout=TDR_LLM_TIMEOUT,
                max_retries=0,
            )
            request_kwargs = {
                "model": TDR_LLM_REMOTE_MODEL,
                "messages": messages,
                "temperature": 0,
                "max_tokens": token_budget,
            }
            request_kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**request_kwargs)
            llm_audit["attempts"] = attempt + 1
            llm_audit["reported_model"] = getattr(response, "model", None)
            usage = getattr(response, "usage", None)
            if usage is not None:
                llm_audit["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
                llm_audit["completion_tokens"] = getattr(usage, "completion_tokens", None)
                llm_audit["total_tokens"] = getattr(usage, "total_tokens", None)
            llm_audit["request_id"] = getattr(response, "id", None)
            if not response.choices:
                raise ValueError("Le serveur LLM n'a renvoyé aucun choix (choices vide).")
            choice = response.choices[0]
            message = choice.message
            last_content = (getattr(message, "content", None) or "").strip()
            finish_reason = getattr(choice, "finish_reason", None)
            llm_audit["finish_reason"] = finish_reason
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
                "TDR LLM tentative %d/2 échouée (modèle=%s, budget=%d) : %s",
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
        f"Le fournisseur LLM ({ACTIVE_MODEL}) n'a pas produit de JSON valide après 2 tentatives. "
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
        # Même extraction pour le candidat et les références afin d'éviter
        # des écarts de traitement entre les deux côtés de la comparaison.
        from src.extraction.extract_docx import extract_docx

        return extract_docx(path)
    raise IndependentTDRError("Format non pris en charge. Utilisez un fichier DOCX ou PDF.")


def _excerpt(text: str, limit: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", (text or "")).strip()
    if len(text) <= limit:
        return text
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    priority = ("objet", "contexte", "objectif", "résultat", "zone", "population",
                "méthodologie", "activité", "livrable", "période", "justification")
    ranked = sorted(
        enumerate(paragraphs),
        key=lambda item: sum(word in item[1].casefold() for word in priority),
        reverse=True,
    )
    selected, used = set(), 0
    for idx, paragraph in ranked:
        remaining = limit - used
        if remaining <= 0:
            break
        piece = paragraph if len(paragraph) <= remaining else paragraph[:remaining]
        if piece:
            selected.add((idx, piece))
            used += len(piece) + 2
    selected_by_index = {idx: piece for idx, piece in selected}
    return "\n\n".join(selected_by_index[i] for i in sorted(selected_by_index))


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
            is_reference=True, document_type="TDR"
        ).filter(Q(cleaned_text__gt="") | Q(extracted_text__gt=""))
        .exclude(embedding=[])
        # Le moteur RAG autonome n'utilise pas les vecteurs JSON stockés en base :
        # il encode les textes du registre et conserve les vecteurs dans son cache
        # mémoire. Ne pas transférer ces gros champs à chaque analyse.
        .only(
            "id",
            "title",
            "file_path",
            "document_type",
            "extracted_text",
            "cleaned_text",
            "content_hash",
            "embedding_model",
        )
        .order_by("id")
    )


def _semantic_chunks(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    """Découpe un TDR en segments qui conservent le contexte local."""
    normalized = " ".join(str(text or "").split())
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _reference_cache_key(rows) -> tuple:
    """Empreinte déterministe du corpus pour invalider le cache si une référence change."""
    fingerprint = []
    for row in rows:
        source = (row.extracted_text or row.cleaned_text or row.title or "").strip()
        digest = str(row.content_hash or hashlib.sha256(source.encode("utf-8")).hexdigest())
        fingerprint.append((row.pk, digest, str(row.embedding_model or "")))
    return tuple(fingerprint)


def _get_reference_search_cache(rows, model, metrics: dict | None = None):
    """Construit une seule fois les représentations coûteuses des références."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    global _REFERENCE_SEARCH_CACHE, _REFERENCE_SEARCH_CACHE_KEY

    cache_key = _reference_cache_key(rows)
    with _REFERENCE_SEARCH_CACHE_LOCK:
        if _REFERENCE_SEARCH_CACHE is not None and _REFERENCE_SEARCH_CACHE_KEY == cache_key:
            if metrics is not None:
                metrics["reference_cache_build_ms"] = 0
            return _REFERENCE_SEARCH_CACHE, True

        cache_started = time.perf_counter()
        reference_texts = [
            (row.cleaned_text or row.extracted_text or row.title or "").strip()
            for row in rows
        ]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        try:
            reference_matrix = vectorizer.fit_transform(reference_texts)
        except ValueError:
            vectorizer = None
            reference_matrix = None

        chunk_texts = []
        chunk_owners = []
        for index, row in enumerate(rows):
            source = row.extracted_text or row.cleaned_text or row.title or ""
            chunks = _semantic_chunks(source) or [str(row.title or "")]
            chunk_texts.extend(chunks)
            chunk_owners.extend([index] * len(chunks))

        chunk_vectors = model.encode(
            chunk_texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        cache = {
            "vectorizer": vectorizer,
            "reference_matrix": reference_matrix,
            "chunk_vectors": np.asarray(chunk_vectors, dtype=np.float32),
            "chunk_owners": np.asarray(chunk_owners, dtype=np.int32),
            "chunk_count": len(chunk_texts),
        }
        _REFERENCE_SEARCH_CACHE = cache
        _REFERENCE_SEARCH_CACHE_KEY = cache_key
        if metrics is not None:
            metrics["reference_cache_build_ms"] = round(
                (time.perf_counter() - cache_started) * 1000
            )
        return cache, False


def _rank_all_references(candidate: str, rows, metrics: dict | None = None):
    """Classe tout le corpus ; le LLM examine intégralement la référence la mieux classée."""
    from sklearn.metrics.pairwise import cosine_similarity
    from src.similarity.model_manager import get_model

    model_started = time.perf_counter()
    model = get_model()
    model_load_ms = round((time.perf_counter() - model_started) * 1000)
    cache, cache_hit = _get_reference_search_cache(rows, model, metrics=metrics)

    if cache["vectorizer"] is not None and cache["reference_matrix"] is not None:
        candidate_tfidf = cache["vectorizer"].transform([candidate])
        lexical_scores = cosine_similarity(
            candidate_tfidf, cache["reference_matrix"]
        ).ravel()
    else:
        lexical_scores = np.zeros(len(rows), dtype=np.float32)

    # Seul le vecteur du document candidat est recalculé à chaque analyse.
    candidate_embedding_started = time.perf_counter()
    candidate_vector = model.encode(
        [candidate], normalize_embeddings=True, show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    candidate_embedding_ms = round(
        (time.perf_counter() - candidate_embedding_started) * 1000
    )
    scoring_started = time.perf_counter()
    similarities = cache["chunk_vectors"] @ np.asarray(candidate_vector, dtype=np.float32)
    document_scores = [[] for _ in rows]
    for owner, score in zip(cache["chunk_owners"], similarities):
        document_scores[int(owner)].append(float(score))

    ranked = []
    for index, row in enumerate(rows):
        best_chunks = sorted(document_scores[index], reverse=True)[:3]
        semantic = float(np.mean(best_chunks)) if best_chunks else 0.0
        semantic = max(0.0, min(1.0, semantic))
        lexical = max(0.0, min(1.0, float(lexical_scores[index])))
        hybrid = 0.30 * lexical + 0.70 * semantic
        ranked.append((hybrid, lexical, semantic, row))
    ranked.sort(key=lambda item: item[0], reverse=True)

    if metrics is not None:
        metrics["model_load_ms"] = model_load_ms
        metrics["candidate_embedding_ms"] = candidate_embedding_ms
        metrics["similarity_scoring_ms"] = round(
            (time.perf_counter() - scoring_started) * 1000
        )
        metrics["reference_embedding_cache_hit"] = cache_hit
        metrics["reference_chunks_cached"] = cache["chunk_count"]
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


def _normalize_evidence(value: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _verified_evidence(values, source_text: str) -> list[str]:
    """Conserve uniquement les citations exactes et assez longues du document source."""
    if not isinstance(values, list):
        return []
    source = _normalize_evidence(source_text)
    verified = []
    for value in values[:8]:
        quote = str(value).strip()[:700]
        normalized = _normalize_evidence(quote)
        if len(normalized) >= 35 and normalized in source and quote not in verified:
            verified.append(quote)
    return verified


def _recover_case2_evidence(
    source_text: str,
    candidate_text: str,
    common_points,
    existing_evidence: list[str],
) -> list[str]:
    """Récupère des phrases exactes de la référence pour étayer un CAS_2.

    Cette récupération ne décide jamais du verdict : elle ne s'exécute qu'après
    le choix CAS_2 du LLM et ne retient que des phrases réellement présentes
    dans le document source, avec plusieurs termes métier partagés.
    """
    import unicodedata

    stopwords = {
        "avec", "dans", "pour", "par", "sur", "les", "des", "une", "dans",
        "entre", "ainsi", "leur", "leurs", "cette", "ces", "sont", "être",
        "avoir", "plus", "moins", "tout", "tous", "elle", "elles", "nous",
        "vous", "qui", "que", "quoi", "dont", "aux", "au", "du", "de",
        "la", "le", "un", "et", "ou", "en", "est", "sa", "son", "ses",
        "comme", "notamment", "également", "etc", "afin", "cadre",
    }

    def terms(value: str) -> set[str]:
        normalized = unicodedata.normalize("NFKD", str(value or "").casefold())
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        return {
            token for token in re.findall(r"[a-z0-9]{4,}", normalized)
            if token not in stopwords
        }

    points = " ".join(str(item) for item in common_points) if isinstance(common_points, list) else ""
    query_terms = terms(points + " " + candidate_text)
    if len(query_terms) < 3:
        return list(existing_evidence[:2])

    recovered = list(existing_evidence[:2])
    normalized_existing = {_normalize_evidence(item) for item in recovered}
    sentences = re.split(r"(?<=[.!?])\s+|\n+", source_text or "")
    ranked = []
    for sentence in sentences:
        quote = sentence.strip()
        sentence_terms = terms(quote)
        if len(quote) < 55 or len(sentence_terms) < 5:
            continue
        normalized = _normalize_evidence(quote)
        if normalized in normalized_existing:
            continue
        overlap = sentence_terms & query_terms
        if len(overlap) < 3:
            continue
        coverage = len(overlap) / max(1, len(sentence_terms))
        if coverage < 0.25:
            continue
        ranked.append((len(overlap), coverage, quote))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for _, _, quote in ranked:
        if len(recovered) >= 2:
            break
        recovered.append(quote)
    return recovered


# ---------------------------------------------------------------------------
# Analyse principale
# ---------------------------------------------------------------------------
def analyze_tdr_independently(candidate_text: str) -> dict:
    """Classe le registre, puis demande au LLM un verdict sur le TDR complet et sa référence la plus proche."""
    analysis_started = time.perf_counter()
    performance = {}
    candidate = (candidate_text or "").strip()
    if len(candidate) < 80:
        raise IndependentTDRError("Le TDR contient trop peu de texte exploitable.")
    load_started = time.perf_counter()
    rows = _load_all_indexed_references()
    performance["reference_load_ms"] = round((time.perf_counter() - load_started) * 1000)
    if not rows:
        raise IndependentTDRError(
            "Aucun TDR de référence n'est présent dans l'index PostgreSQL. "
            "Préparez l'index avant de lancer l'analyse."
        )

    # Le classement sert uniquement à retrouver les références candidates ;
    # le LLM reste responsable du verdict métier.
    retrieval_started = time.perf_counter()
    ranked = _rank_all_references(candidate, rows, metrics=performance)
    performance["retrieval_ms"] = round((time.perf_counter() - retrieval_started) * 1000)
    if not ranked:
        raise IndependentTDRError("Aucune référence TDR exploitable n'a été trouvée.")
    # Le classement parcourt tout le registre, mais l'arbitrage détaillé porte
    # sur la seule référence la mieux classée. Les scores ne sont jamais des
    # seuils de décision métier.
    selected = ranked[:1]
    candidate_excerpt = candidate

    reference_blocks = []
    reference_excerpt_chars = 0
    for _, _, _, row in selected:
        # Le LLM reçoit le texte intégral de la référence, sans troncature.
        reference_excerpt = row.extracted_text or row.cleaned_text or ""
        reference_excerpt_chars += len(reference_excerpt)
        reference_blocks.append(
            "DOCUMENT DE RÉFÉRENCE\n"
            f"ID: {row.id}\n"
            f"TITRE: {row.title}\n"
            f"TYPE DE DOCUMENT: {row.get_document_type_display()}\n"
            f"FICHIER: {Path(row.file_path).name}\n"
            f"CONTENU:\n{reference_excerpt}"
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
    performance["prompt_build_ms"] = round(
        (time.perf_counter() - retrieval_started) * 1000
    ) - performance["retrieval_ms"]
    performance["candidate_chars"] = len(candidate)
    performance["candidate_excerpt_chars"] = len(candidate_excerpt)
    performance["reference_excerpt_chars"] = reference_excerpt_chars
    performance["prompt_chars"] = len(SYSTEM_PROMPT) + len(user_prompt)
    llm_started = time.perf_counter()
    final = _request_llm_json(SYSTEM_PROMPT, user_prompt, tokens=OUTPUT_TOKENS)
    performance["llm_ms"] = round((time.perf_counter() - llm_started) * 1000)
    validation_started = time.perf_counter()

    llm_audit = final.pop("_llm_audit", {
        "provider": TDR_LLM_PROVIDER,
        "requested_model": ACTIVE_MODEL,
        "reported_model": None,
        "request_id": None,
    })

    case_id = str(final.get("case_id", CASE_3))
    if case_id not in CASES:
        logger.warning("case_id inconnu renvoyé par le modèle : %r -> vérification requise", case_id)
        raise IndependentTDRError(
            "Analyse insuffisante : le modèle a renvoyé un verdict non reconnu. "
            "Aucun verdict n'a été enregistré."
        )
    if str(final.get("decision", "")).strip().upper() == "ANALYSE_INSUFFISANTE":
        raise InsufficientEvidenceError(
            "Analyse insuffisante : les éléments disponibles ne permettent pas de conclure. "
            "Vérifiez les objectifs, la zone/population, la méthodologie et les livrables."
        )

    # Le verdict ne doit jamais être promu par détection de mots dans l'explication.
    # Les preuves documentaires seront vérifiées après identification de la référence.

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
            row = None  # Un nom approximatif ne constitue pas une référence autorisée.
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

    # Une citation doit être réellement présente dans le texte de la référence.
    matched_row = matched_rows[0] if matched_rows else None
    model_item = _model_item_for(matched_row) if matched_row is not None else {}
    source_text = (
        matched_row.extracted_text or matched_row.cleaned_text or ""
        if matched_row is not None else ""
    )
    source_evidence = _verified_evidence(
        model_item.get("source_evidence") or final.get("evidence_from_reference", []),
        source_text,
    )

    # CAS_2 : si le modèle a fourni trop peu de citations, récupérer deux
    # phrases exactes dans la référence retenue à partir des points communs
    # qu'il a lui-même identifiés. Cette étape ne modifie jamais le verdict.
    if case_id == CASE_2 and matched_row is not None and len(source_evidence) < 2:
        recovery_points = (
            model_item.get("common_points")
            or final.get("common_points", [])
        )
        source_evidence = _recover_case2_evidence(
            source_text=source_text,
            candidate_text=candidate,
            common_points=recovery_points,
            existing_evidence=source_evidence,
        )

    # Aucun CAS_1/CAS_2 sans référence exacte et au moins deux citations vérifiées.
    if case_id in {CASE_1, CASE_2} and (
        matched_row is None or len(source_evidence) < 2
    ):
        raise InsufficientEvidenceError(
            "Analyse insuffisante : la référence retenue ou les deux extraits exacts "
            "nécessaires n'ont pas pu être vérifiés dans le document source. "
            "Aucun verdict de doublon n'a été enregistré. Vérifiez l'extraction et l'index, "
            "puis relancez l'analyse."
        )

    if case_id == CASE_3:
        # Un CAS_3 accompagné d'une référence ou de citations est contradictoire.
        # Ne pas le promouvoir automatiquement : demander une vérification humaine.
        if matched_row is not None or source_evidence or final.get("evidence_from_reference"):
            raise InsufficientEvidenceError(
                "Analyse insuffisante : le verdict « aucune correspondance » contredit "
                "la référence ou les extraits cités par le modèle. Vérification humaine requise."
            )
        matched_row = None
        matched_rows = []
        source_evidence = []
    else:
        matched_rows = [matched_row]

    reference_name = Path(matched_row.file_path).name if matched_row else ""
    # La recommandation est produite côté serveur à partir de la référence
    # effectivement validée. Cela évite qu'un alias ou un ancien identifiant
    # inventé par le LLM contredise le fichier affiché à l'utilisateur.
    reference_label = Path(reference_name).stem if reference_name else ""
    if case_id == CASE_1 and reference_label:
        recommendation = (
            f"Considérer le TDR candidat comme un doublon ou une reformulation "
            f"de la référence « {reference_label} » du registre."
        )
    elif case_id == CASE_2 and reference_label:
        recommendation = (
            f"Faire vérifier les éléments communs avec la référence « {reference_label} » "
            "avant toute validation ou programmation de l'étude."
        )
    else:
        recommendation = (
            "Aucune référence du registre ne présente de correspondance métier suffisante. "
            "Le TDR peut poursuivre le circuit normal de validation."
        )

    per_reference = []
    if matched_row is not None:
        level = "forte" if case_id == CASE_1 else "partielle"
        per_reference.append({
            "source_id": matched_row.id,
            "source": reference_name,
            "title": str(matched_row.title or ""),
            "document_type": matched_row.get_document_type_display(),
            "level": level,
            "explanation": str(model_item.get("explanation") or final.get("explanation", ""))[:1200],
            "common_points": _as_list(
                model_item.get("common_points") or final.get("common_points", []), 8, 400),
            "differences": _as_list(
                model_item.get("differences") or final.get("differences", []), 8, 400),
            "source_evidence": source_evidence,
        })

    evidence = [{
        "source": item["source"],
        "point": quote,
        "justification": "Extrait vérifié : le texte cité est présent dans le document de référence.",
    } for item in per_reference for quote in item["source_evidence"]]

    # L'interface ne présente qu'une référence : retirer de l'explication
    # toute phrase qui évoque les candidates écartées ou une comparaison de liste.
    raw_explanation = str(final.get("explanation", "")).strip()
    excluded_names = [
        Path(row.file_path).name for _, _, _, row in selected
        if matched_row is None or row.id != matched_row.id
    ]
    explanation_sentences = re.split(r"(?<=[.!?])\s+", raw_explanation)
    visible_sentences = []
    for sentence in explanation_sentences:
        folded = sentence.casefold()
        if any(name.casefold() in folded for name in excluded_names):
            continue
        if any(term in folded for term in (
            "les autres références", "les autres documents",
            "références écartées", "moins pertinentes",
        )):
            continue
        if sentence.strip():
            visible_sentences.append(sentence.strip())
    display_explanation = " ".join(visible_sentences)

    performance["validation_and_formatting_ms"] = round(
        (time.perf_counter() - validation_started) * 1000
    )
    performance["total_ms"] = round(
        (time.perf_counter() - analysis_started) * 1000
    )

    return {
        "mode": "tdr_rag_independent",
        "decision_provider": "independent_rag",
        "decision": case_id,
        "rag_analysis": {
            "case_id": case_id,
            "case_title": CASE_TITLES[case_id],
            "model": llm_audit.get("reported_model") or ACTIVE_MODEL,
            "model_audit": llm_audit,
            "decision": (
                "Correspondance forte" if case_id == CASE_1 else
                "Correspondance partielle" if case_id == CASE_2 else
                "Aucune correspondance"
            ),
            "explanation": display_explanation[:2500],
            "matched_reference": reference_name,
            "matched_references": per_reference,
            "confidence": "À confirmer",
            "common_points": _as_list(final.get("common_points", []), 10, 500),
            "differences": _as_list(final.get("differences", []), 10, 500),
            "recommendation": recommendation,
            "correspondence_level": (
                "forte" if case_id == CASE_1 else
                "moyenne" if case_id == CASE_2 else "nulle"
            ),
            "evidence_from_reference": source_evidence,
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
            "performance": dict(performance),
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
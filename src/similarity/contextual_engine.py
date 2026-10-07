"""Analyse contextuelle des documents pour TDRDOC-SCAN.

Le moteur utilise le modèle Sentence-Transformers déjà présent dans le projet.
Il fournit une preuve contextuelle indépendante après l'identification des
passages proches, sans modifier le TF-IDF ni le score sémantique des passages.
"""
import re
import unicodedata

import numpy as np

from src.similarity.model_manager import get_model


CONTEXT_DIMENSIONS = (
    "objet",
    "zone",
    "periode",
    "population",
    "objectifs",
    "resultats",
    "methodologie",
)

_LABELS = {
    "objet": ("objet", "theme", "thème", "problematique", "problématique"),
    "zone": ("zone", "localisation", "territoire", "champ geographique", "champ géographique"),
    "periode": ("periode", "période", "duree", "durée", "calendrier", "horizon"),
    "population": ("population", "beneficiaires", "bénéficiaires", "public cible", "groupe cible"),
    "objectifs": ("objectif", "objectifs", "but", "finalite", "finalité"),
    "resultats": ("resultat", "résultat", "resultats", "résultats", "livrable", "produit attendu"),
    "methodologie": ("methodologie", "méthodologie", "approche", "methode", "méthode", "dispositif"),
}


def _normalize(text):
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().lower()


def _prepare_text(text):
    """Réintroduit des séparateurs lorsque l'extraction DOCX les a perdus."""
    text = text or ""
    pattern = (
        r"(?i)\\s+(?=(?:"
        r"\\d+(?:\\.\\d+)*\\s*[.)-]\\s+|"
        r"(?:contexte|justification|objectifs?|champ de l[’']étude|"
        r"résultats? attendus?|méthodologie|approche|population|"
        r"zone|localisation|livrables?|produits?|durée|calendrier|"
        r"chronogramme|collecte|analyse|conclusion)\\b"
        r"))"
    )
    return re.sub(pattern, "\\n", text)


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


_DIMENSION_CUES = {
    "objet": (
        "etude portant", "étude portant", "etude consacree", "étude consacrée",
        "etude sur", "étude sur", "diagnostic de", "diagnostic sur",
        "mission porte sur", "mission consacree", "mission consacrée",
        "objet de l'etude", "objet de l’étude", "theme de l'etude", "thème de l’étude",
    ),
    "zone": (
        "zone", "zones", "localisation", "territoire", "territoires",
        "province", "provinces", "commune", "communes", "region", "région",
        "localite", "localités", "localites", "champ geographique", "champ géographique",
        "dans le ", "dans les ", "au niveau de ", "au ", "aux ",
    ),
    "periode": (
        "annee", "année", "periode", "période", "duree", "durée",
        "calendrier", "chronogramme", "jours", "semaine", "mois",
    ),
    "population": (
        "population cible", "public cible", "groupe cible", "beneficiaires",
        "bénéficiaires", "acteurs concernés", "acteurs concernes", "menages",
        "ménages", "producteurs", "structures ciblees", "structures ciblées",
    ),
    "objectifs": (
        "objectif general", "objectif général", "objectifs specifiques",
        "objectifs spécifiques", "objectif spécifique", "but de l'etude",
        "but de l’étude", "finalite de l'etude", "finalité de l’étude",
    ),
    "resultats": (
        "resultats attendus", "résultats attendus", "produits attendus",
        "livrables", "produits livrables", "permettra de produire",
    ),
    "methodologie": (
        "methodologie", "méthodologie", "approche méthodologique",
        "approche retenue", "methode", "méthode", "collecte de donnees",
        "collecte de données", "revue documentaire",
    ),
}


def _extract_dimension(text, dimension):
    prepared = _prepare_text(text)
    lines = [line.strip() for line in re.split(r"\n+", prepared) if line.strip()]
    labels = tuple(_normalize(x) for x in _LABELS[dimension])
    hits = []

    for index, line in enumerate(lines):
        folded = _normalize(line)
        if any(re.search(rf"\b{re.escape(label)}\b", folded) for label in labels):
            chunk = " ".join(lines[index:index + 4])
            if len(chunk) >= 35:
                hits.append(chunk)

    if hits:
        return " ".join(hits)[:1800]

    selected = []
    cues = tuple(_normalize(x) for x in _DIMENSION_CUES[dimension])
    for sentence in _sentences(prepared):
        folded = _normalize(sentence)
        if (
            any(re.search(rf"\b{re.escape(label)}\b", folded) for label in labels)
            or any(cue in folded for cue in cues)
        ):
            selected.append(sentence)

    if dimension == "objet":
        selected = selected[:6]
        if not selected:
            selected = _sentences(prepared)[:4]

    return " ".join(selected)[:1800]


def extract_context(text):
    context = {dimension: _extract_dimension(text, dimension) for dimension in CONTEXT_DIMENSIONS}

    zone_sentences = []
    for sentence in _sentences(_prepare_text(text)):
        folded = _normalize(sentence)
        if any(cue in folded for cue in _DIMENSION_CUES["zone"]):
            zone_sentences.append(sentence)
    if zone_sentences:
        context["zone"] = " ".join(zone_sentences)[:1800]

    # L'identité de l'étude ne doit pas dépendre uniquement de titres de
    # sections. Les TDR réels sont parfois extraits en texte continu.
    context["identite_etude"] = (text or "")[:3000]

    return context


def _semantic_similarity(model, left, right):
    if not left or not right:
        return None
    vectors = model.encode(
        [left, right],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return float(np.clip(np.dot(vectors[0], vectors[1]), 0.0, 1.0))


def analyze_context(candidate_text, source_text):
    """Compare les dimensions contextuelles sans modifier les scores existants."""
    candidate = extract_context(candidate_text)
    source = extract_context(source_text)
    model = get_model()
    dimensions = {}
    available_scores = []

    identity_score = _semantic_similarity(
        model,
        candidate["identite_etude"],
        source["identite_etude"],
    )
    identity_score = identity_score if identity_score is not None else 0.0

    for dimension in CONTEXT_DIMENSIONS:
        score = _semantic_similarity(model, candidate[dimension], source[dimension])
        if score is None:
            dimensions[dimension] = {"score": None, "status": "NON_DETERMINE"}
            continue
        available_scores.append(score)
        dimensions[dimension] = {
            "score": round(score, 4),
            "status": (
                "SIMILAIRE" if score >= 0.68
                else "DIFFERENT" if score < 0.45
                else "INCERTAIN"
            ),
        }

    if not available_scores:
        return {"score": 0.0, "verdict": "À EXAMINER", "confidence": "FAIBLE", "dimensions": dimensions}

    context_score = float(np.mean(available_scores))

    critical = [
        dimensions[name]["score"]
        for name in ("objet", "objectifs", "resultats")
        if dimensions[name]["score"] is not None
    ]
    critical_mean = float(np.mean(critical)) if critical else context_score

    # L'identité globale de l'étude est une preuve contextuelle supplémentaire.
    # Elle est particulièrement importante lorsque le DOCX a perdu ses
    # retours à la ligne pendant l'extraction.
    combined_score = (
        0.35 * identity_score
        + 0.30 * critical_mean
        + 0.35 * context_score
    )

    if identity_score >= 0.80 and critical_mean >= 0.62:
        verdict = "SIMILAIRE"
        confidence = "FORTE"
    elif combined_score >= 0.68 and critical_mean >= 0.55:
        verdict = "SIMILAIRE"
        confidence = "MOYENNE"
    elif identity_score < 0.52 and critical_mean < 0.45:
        verdict = "DIFFERENT"
        confidence = "FORTE"
    elif combined_score < 0.48:
        verdict = "DIFFERENT"
        confidence = "MOYENNE"
    else:
        verdict = "À EXAMINER"
        confidence = "MOYENNE"

    return {
        "score": round(combined_score, 4),
        "identity_score": round(identity_score, 4),
        "verdict": verdict,
        "confidence": confidence,
        "dimensions": dimensions,
    }

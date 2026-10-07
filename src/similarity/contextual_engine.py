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


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def _extract_dimension(text, dimension):
    lines = [line.strip() for line in re.split(r"\n+", text or "") if line.strip()]
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
    for sentence in _sentences(text):
        folded = _normalize(sentence)
        if any(re.search(rf"\b{re.escape(label)}\b", folded) for label in labels):
            selected.append(sentence)

    return " ".join(selected)[:1800]


def extract_context(text):
    return {dimension: _extract_dimension(text, dimension) for dimension in CONTEXT_DIMENSIONS}


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
        for name in ("objet", "objectifs")
        if dimensions[name]["score"] is not None
    ]
    critical_mean = float(np.mean(critical)) if critical else context_score

    if context_score >= 0.68 and critical_mean >= 0.60:
        verdict = "SIMILAIRE"
        confidence = "FORTE" if context_score >= 0.78 else "MOYENNE"
    elif context_score < 0.45 or critical_mean < 0.42:
        verdict = "DIFFERENT"
        confidence = "FORTE" if context_score < 0.35 else "MOYENNE"
    else:
        verdict = "À EXAMINER"
        confidence = "MOYENNE"

    return {
        "score": round(context_score, 4),
        "verdict": verdict,
        "confidence": confidence,
        "dimensions": dimensions,
    }

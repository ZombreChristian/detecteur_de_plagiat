"""Analyse contextuelle des documents pour TDRDOC-SCAN.

Le moteur compare l'identité d'une étude et ses dimensions métier. Il reste
indépendant du TF-IDF, du matching des passages et du seuil de décision.
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

# Les quatre premières dimensions décrivent l'identité de l'étude.
# La méthodologie, le calendrier et les éléments administratifs sont moins
# discriminants : deux TDR peuvent partager la même structure sans être le
# même sujet.
CONTEXT_WEIGHTS = {
    "objet": 0.27,
    "zone": 0.24,
    "objectifs": 0.20,
    "resultats": 0.15,
    "population": 0.07,
    "periode": 0.04,
    "methodologie": 0.03,
}

_LABELS = {
    "objet": ("objet", "theme", "thème", "problematique", "problématique"),
    "zone": ("zone", "localisation", "territoire", "champ geographique", "champ géographique"),
    "periode": ("periode", "période", "duree", "durée", "calendrier", "horizon"),
    "population": ("population", "beneficiaires", "bénéficiaires", "public cible", "groupe cible"),
    "objectifs": ("objectif", "objectifs", "but", "finalite", "finalité"),
    "resultats": ("resultat", "résultat", "resultats", "résultats", "livrable", "produit attendu"),
    "methodologie": ("methodologie", "méthodologie", "approche", "methode", "méthode", "dispositif"),
}

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
        "localite", "localités", "localites", "département", "departement",
        "champ geographique", "champ géographique",
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


def _normalize(text):
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().lower()


def _prepare_text(text):
    """Réintroduit des séparateurs lorsque l'extraction DOCX les a perdus."""
    text = text or ""
    pattern = (
        r"(?i)\s+(?=(?:"
        r"\d+(?:\.\d+)*\s*[.)-]\s+|"
        r"(?:contexte|justification|objectifs?|champ de l[’']étude|"
        r"résultats? attendus?|méthodologie|approche|population|"
        r"zone|localisation|livrables?|produits?|durée|calendrier|"
        r"chronogramme|collecte|analyse|conclusion)\b"
        r"))"
    )
    return re.sub(pattern, "\n", text)


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


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
        # L'objet doit représenter le sujet de l'étude, pas le contexte
        # administratif commun à tous les TDR.
        title_lines = lines[:2]
        title_text = " ".join(title_lines).strip()
        selected = selected[:8]
        if title_text:
            selected = [title_text] + selected
        selected = list(dict.fromkeys(selected))

    return " ".join(selected)[:1800]


def extract_context(text):
    prepared = _prepare_text(text)
    context = {
        dimension: _extract_dimension(prepared, dimension)
        for dimension in CONTEXT_DIMENSIONS
    }

    # La zone est particulièrement sensible : on ne considère plus des mots
    # génériques comme "au" ou "dans le" comme des indices géographiques.
    zone_sentences = []
    for sentence in _sentences(prepared):
        folded = _normalize(sentence)
        if any(cue in folded for cue in _DIMENSION_CUES["zone"]):
            zone_sentences.append(sentence)
    if zone_sentences:
        context["zone"] = " ".join(zone_sentences)[:1800]

    # Pour l'identité globale, on compare le début du TDR mais aussi les
    # sections métier extraites. Cela évite qu'un long bloc méthodologique
    # domine toute la représentation.
    identity_parts = [
        context["objet"],
        context["zone"],
        context["objectifs"],
        context["resultats"],
    ]
    identity_text = " ".join(part for part in identity_parts if part).strip()
    context["identite_etude"] = identity_text[:5000] or (text or "")[:3000]

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


def _dimension_status(score):
    if score is None:
        return "NON_DETERMINE"
    if score >= 0.68:
        return "SIMILAIRE"
    if score < 0.45:
        return "DIFFERENT"
    return "INCERTAIN"


def _specific_tokens(text):
    """Extrait les termes porteurs de sens sans vocabulaire métier codé en dur."""
    normalized = _normalize(text)
    tokens = set(re.findall(r"[a-z]{4,}", normalized))
    generic = {
        "etude", "etudes", "objectif", "objectifs", "objet", "diagnostic",
        "analyse", "analyses", "methodologie", "donnees", "resultats",
        "recommandation", "recommandations", "rapport", "mission",
        "consultant", "consultants", "prestation", "prestations",
        "livrable", "livrables", "suivi", "evaluation", "contraintes",
        "contexte", "besoins", "proposition", "activites", "travaux",
        "document", "documents", "information", "informations", "terrain",
        "collecte", "produire", "production", "validation", "comite",
        "comite", "parties", "prenantes", "reunions", "resultat",
        "provisoire", "conclusion", "cadre", "programme", "programmes",
        "interventions", "acteurs", "structures", "localites", "zones",
    }
    return tokens - generic


def _specific_overlap(left, right):
    left_tokens = _specific_tokens(left)
    right_tokens = _specific_tokens(right)
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return None
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def analyze_context(candidate_text, source_text):
    """Compare les dimensions contextuelles sans modifier les scores existants."""
    candidate = extract_context(candidate_text)
    source = extract_context(source_text)
    model = get_model()
    dimensions = {}

    for dimension in CONTEXT_DIMENSIONS:
        score = _semantic_similarity(model, candidate[dimension], source[dimension])
        dimensions[dimension] = {
            "score": round(score, 4) if score is not None else None,
            "status": _dimension_status(score),
        }

    available = {
        name: data["score"]
        for name, data in dimensions.items()
        if data["score"] is not None
    }
    if not available:
        return {
            "score": 0.0,
            "verdict": "À EXAMINER",
            "confidence": "FAIBLE",
            "dimensions": dimensions,
        }

    weight_total = sum(CONTEXT_WEIGHTS[name] for name in available)
    context_score = sum(
        CONTEXT_WEIGHTS[name] * score for name, score in available.items()
    ) / weight_total

    critical_names = ("objet", "zone", "objectifs", "resultats")
    critical_scores = [
        dimensions[name]["score"]
        for name in critical_names
        if dimensions[name]["score"] is not None
    ]
    critical_mean = float(np.mean(critical_scores)) if critical_scores else context_score

    identity_score = _semantic_similarity(
        model,
        candidate["identite_etude"],
        source["identite_etude"],
    )
    identity_score = identity_score if identity_score is not None else 0.0

    # Les dimensions d'identité dominent. La méthodologie et les éléments
    # administratifs ne peuvent donc plus faire monter fortement le verdict.
    combined_score = (
        0.55 * context_score
        + 0.30 * identity_score
        + 0.15 * critical_mean
    )

    # Une divergence forte sur l'objet ou la zone est une preuve contre le
    # doublon, même si la structure du TDR est très proche.
    identity_conflicts = [
        dimensions[name]["score"]
        for name in ("objet", "zone")
        if dimensions[name]["score"] is not None
    ]

    # Une forte proximité de rédaction ne suffit pas si les termes
    # spécifiques de l'objet ou de la zone ne se recouvrent pratiquement pas.
    # Cette règle est générique : aucun secteur, produit ou ville n'est codé.
    object_overlap = _specific_overlap(candidate["objet"], source["objet"])
    zone_overlap = _specific_overlap(candidate["zone"], source["zone"])

    specific_identity_conflict = (
        (object_overlap is not None and object_overlap < 0.12)
        or (zone_overlap is not None and zone_overlap < 0.12)
    )
    strong_identity_conflict = (
        any(score < 0.45 for score in identity_conflicts)
        or specific_identity_conflict
    )

    if strong_identity_conflict:
        verdict = "DIFFERENT"
        confidence = "FORTE"
    elif (
        identity_score >= 0.80
        and context_score >= 0.68
        and critical_mean >= 0.62
    ):
        verdict = "SIMILAIRE"
        confidence = "FORTE"
    elif (
        combined_score >= 0.68
        and context_score >= 0.62
        and critical_mean >= 0.55
    ):
        verdict = "SIMILAIRE"
        confidence = "MOYENNE"
    elif combined_score < 0.48 or (
        identity_score < 0.52 and critical_mean < 0.45
    ):
        verdict = "DIFFERENT"
        confidence = "MOYENNE"
    else:
        verdict = "À EXAMINER"
        confidence = "MOYENNE"

    return {
        "score": round(float(combined_score), 4),
        "identity_score": round(float(identity_score), 4),
        "verdict": verdict,
        "confidence": confidence,
        "dimensions": dimensions,
    }

"""Profil de couverture du contenu par rapport au corpus existant."""
from __future__ import annotations
import re
from typing import Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing.clean_text import clean_text
from src.passages.passage_engine import adjusted_hybrid_score
from src.similarity.model_manager import get_model

COMPONENTS = [
    ("contexte_justification", "Contexte / justification", r"contexte|justification|situation|problématique"),
    ("objet", "Objet de l'étude", r"objet de l.?étude|objet de l.?etude|objet"),
    ("objectifs", "Objectifs", r"objectifs?|objectif général|objectif specifique|objectifs spécifiques"),
    ("perimetre", "Périmètre / zone / population", r"périmètre|perimetre|zone géographique|zone d.?étude|population|bénéficiaires|beneficiaires"),
    ("resultats", "Résultats attendus", r"résultats? attendus?|resultats? attendus?|livrables|produits attendus"),
    ("methodologie", "Méthodologie", r"méthodologie|methodologie|approche méthodologique|approche methodologique|méthode|methode"),
    ("planning", "Planning / durée", r"planning|chronogramme|calendrier|durée|duree|délai|delai"),
    ("budget", "Budget", r"budget|coût|cout|financement"),
    ("gouvernance", "Gouvernance / équipe", r"équipe|equipe|comité|comite|pilotage|gouvernance|coordination"),
    ("autres", "Autres contenus", r".*"),
]

def _paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n{2,}|(?<=[.!?])\s+", text or "")
    return [re.sub(r"\s+", " ", b).strip() for b in blocks if len(re.sub(r"\s+", " ", b).strip()) >= 35]

def _component_for_text(text: str) -> tuple[str, str]:
    low = text.casefold()
    for key, label, pattern in COMPONENTS[:-1]:
        if re.search(pattern, low, flags=re.I):
            return key, label
    return "autres", "Autres contenus"

def _split_components(text: str) -> dict[str, dict[str, Any]]:
    paragraphs = _paragraphs(text)
    groups = {key: {"key": key, "label": label, "segments": []} for key, label, _ in COMPONENTS}
    current = "autres"
    lines = [re.sub(r"\s+", " ", x).strip() for x in re.split(r"\n", text or "") if x.strip()]
    for line in lines:
        if len(line) <= 100 and re.search(r"^(?:[0-9]+[.)-]?\s*)?[A-Za-zÀ-ÿ /'’&-]{3,60}$", line):
            key, _ = _component_for_text(line)
            if key != "autres":
                current = key
                continue
        if len(line) >= 20:
            groups[current]["segments"].append(line)
    if not any(g["segments"] for g in groups.values() if g["key"] != "autres"):
        groups = {key: {"key": key, "label": label, "segments": []} for key, label, _ in COMPONENTS}
        for paragraph in paragraphs:
            key, _ = _component_for_text(paragraph)
            groups[key]["segments"].append(paragraph)
    return groups

def _score_segments(segments, rows, embeddings, vectorizer, reference_matrix, lexical_weight, semantic_weight):
    if not segments:
        return []
    cleaned = [clean_text(s) for s in segments]
    valid = [(i, s) for i, s in enumerate(cleaned) if s]
    if not valid:
        return []
    texts = [s for _, s in valid]
    lexical_all = cosine_similarity(vectorizer.transform(texts), reference_matrix)
    segment_embeddings = get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True, batch_size=32)
    semantic_all = np.clip(np.matmul(segment_embeddings, embeddings.T), 0.0, 1.0)
    output = []
    for row_index, (original_index, cleaned_segment) in enumerate(valid):
        best_idx = int(np.argmax(np.maximum(lexical_all[row_index], semantic_all[row_index])))
        lexical = float(lexical_all[row_index, best_idx])
        semantic = float(semantic_all[row_index, best_idx])
        score = float(adjusted_hybrid_score(cleaned_segment, rows[best_idx].extracted_text or rows[best_idx].cleaned_text, lexical, semantic, lexical_weight, semantic_weight))
        output.append({"text": segments[original_index], "length": len(segments[original_index]), "score": round(score, 4), "source": rows[best_idx].title or rows[best_idx].file_path})
    return output

def _classify(score: float, threshold: float):
    partial_threshold = max(0.35, threshold * 0.78)
    if score >= threshold:
        return "EXISTANT"
    if score >= partial_threshold:
        return "PARTIELLEMENT_EXISTANT"
    return "NOUVEAU"

def _weighted_metrics(segments, threshold):
    total = sum(x["length"] for x in segments) or 1
    existing = sum(x["length"] for x in segments if _classify(x["score"], threshold) == "EXISTANT")
    partial = sum(x["length"] for x in segments if _classify(x["score"], threshold) == "PARTIELLEMENT_EXISTANT")
    new = sum(x["length"] for x in segments if _classify(x["score"], threshold) == "NOUVEAU")
    return {"existing": 100 * existing / total, "partial": 100 * partial / total, "new": 100 * new / total}

def build_content_profile(text: str, rows, embeddings, lexical_weight: float, semantic_weight: float, threshold: float, kind: str):
    cleaned = clean_text(text)
    if not cleaned or not rows:
        return {"overall": {"existing_percent": 0.0, "new_percent": 100.0, "partial_percent": 0.0, "segments": 0}, "components": [], "interpretation": {"level": "INSUFFISANT", "recommendation": "Le contenu exploitable est insuffisant pour interpréter la nouveauté."}}
    corpus = [r.cleaned_text for r in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    reference_matrix = vectorizer.fit_transform(corpus)
    groups = _split_components(text)
    scored = _score_segments(_paragraphs(text), rows, embeddings, vectorizer, reference_matrix, lexical_weight, semantic_weight)
    metrics = _weighted_metrics(scored, threshold)
    scored_by_text = {x["text"]: x for x in scored}
    components = []
    for key, label, _ in COMPONENTS:
        segments = [scored_by_text[s] for s in groups[key]["segments"] if s in scored_by_text]
        if not segments:
            continue
        cm = _weighted_metrics(segments, threshold)
        strongest = max(segments, key=lambda x: x["score"])
        components.append({"key": key, "label": label, "existing_percent": round(cm["existing"], 1), "new_percent": round(cm["new"], 1), "partial_percent": round(cm["partial"], 1), "score": round(strongest["score"], 3), "segments": len(segments), "best_source": strongest["source"]})
    effective_existing = metrics["existing"] + 0.5 * metrics["partial"]
    if kind == "duplicate":
        if effective_existing >= 70:
            level, recommendation = "FORTE_PROXIMITE", "Une part importante du TDR correspond au corpus existant. Recommandation : SIMILAIRE, avec examen humain si le contexte institutionnel l'exige."
        elif effective_existing >= 40:
            level, recommendation = "PROXIMITE_MOYENNE", "Une part substantielle du TDR correspond au corpus. Recommandation : À EXAMINER avant acceptation."
        else:
            level, recommendation = "FAIBLE_PROXIMITE", "La majorité du contenu ne présente pas de correspondance forte détectable. Recommandation : DIFFERENT sous réserve du contexte."
    else:
        if effective_existing >= 60:
            level, recommendation = "FORTE_CORRESPONDANCE", "Une part importante du rapport présente une correspondance détectable. Vérifier les passages et les sources avant toute conclusion de plagiat."
        elif effective_existing >= 30:
            level, recommendation = "CORRESPONDANCE_MODEREE", "Des correspondances détectables existent. Une vérification humaine des passages signalés est recommandée."
        else:
            level, recommendation = "FAIBLE_CORRESPONDANCE", "Peu de contenu présente une correspondance forte détectable avec le corpus."
    return {"overall": {"existing_percent": round(metrics["existing"], 1), "new_percent": round(metrics["new"], 1), "partial_percent": round(metrics["partial"], 1), "effective_existing_percent": round(effective_existing, 1), "segments": len(scored)}, "components": components, "interpretation": {"level": level, "recommendation": recommendation, "note": "« Existant » signifie correspondance détectable avec le corpus, et non preuve de plagiat. Les catégories forte, partielle et nouvelle sont pondérées par longueur de contenu."}}

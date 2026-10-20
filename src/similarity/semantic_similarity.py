"""Fusionne TF-IDF et embeddings puis calcule le score hybride offline."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.similarity.model_manager import get_model


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "donnees/documents_nettoyes.csv"
PAIRS = ROOT / "donnees/pairs.csv"
OUT = ROOT / "resultats/semantic_predictions.csv"

LEXICAL_WEIGHT = 0.30
SEMANTIC_WEIGHT = 0.70


def main():
    docs = pd.read_csv(DOCS).set_index("document_id")
    pairs = pd.read_csv(PAIRS)

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    tf = vec.fit_transform(docs.text.fillna(""))
    pos = {d: i for i, d in enumerate(docs.index)}

    pairs["tfidf_score"] = [
        float(cosine_similarity(tf[pos[r.doc_a]], tf[pos[r.doc_b]])[0, 0])
        for _, r in pairs.iterrows()
    ]

    model = get_model()
    emb = model.encode(
        docs.text.fillna("").tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    epos = {d: i for i, d in enumerate(docs.index)}

    pairs["semantic_score"] = [
        float(np.clip(np.dot(emb[epos[r.doc_a]], emb[epos[r.doc_b]]), 0.0, 1.0))
        for _, r in pairs.iterrows()
    ]
    pairs["hybrid_score"] = (
        LEXICAL_WEIGHT * pairs.tfidf_score
        + SEMANTIC_WEIGHT * pairs.semantic_score
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(pairs.groupby("label")[["tfidf_score", "semantic_score", "hybrid_score"]].mean())


if __name__ == "__main__":
    main()

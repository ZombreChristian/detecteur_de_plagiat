"""Génère un export local des embeddings du corpus.

La préparation officielle du registre passe désormais par :
    python frontend/manage.py prepare_reference_index

Ce script reste disponible pour les traitements offline et réutilise le même
modèle local que l'API.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.similarity.model_manager import get_model


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "donnees" / "documents_nettoyes.csv"
OUT = ROOT / "modeles" / "embeddings.npz"


def main():
    docs = pd.read_csv(DOCS)
    model = get_model()
    embeddings = model.encode(
        docs["text"].fillna("").tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        ids=docs["document_id"].astype(str).values,
        embeddings=embeddings,
    )
    print(f"Embeddings créés : {len(embeddings)} documents")


if __name__ == "__main__":
    main()

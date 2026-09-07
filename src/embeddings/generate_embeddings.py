"""Génère et sauvegarde les embeddings Sentence Transformer."""
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "donnees" / "documents_extraits.csv"
OUT = ROOT / "modeles" / "embeddings.npz"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def main():
    docs = pd.read_csv(DOCS)
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(docs["text"].fillna("").tolist(), normalize_embeddings=True, show_progress_bar=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, ids=docs["document_id"].astype(str).values, embeddings=embeddings)
    print(f"Embeddings créés : {len(embeddings)} documents")

if __name__ == "__main__": main()

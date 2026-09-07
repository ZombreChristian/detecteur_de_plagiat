"""Baseline TF-IDF + similarité cosinus."""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "donnees" / "documents_extraits.csv"
PAIRS = ROOT / "donnees" / "pairs.csv"
OUT = ROOT / "resultats" / "tfidf_predictions.csv"

def main():
    docs = pd.read_csv(DOCS).set_index("document_id")
    pairs = pd.read_csv(PAIRS)
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=1, max_df=0.98, sublinear_tf=True)
    X = vectorizer.fit_transform(docs["text"].fillna(""))
    ids = list(docs.index)
    pos = {d:i for i,d in enumerate(ids)}
    scores=[]
    for _, r in pairs.iterrows():
        a,b=r.doc_a,r.doc_b
        score=float(cosine_similarity(X[pos[a]], X[pos[b]])[0,0])
        scores.append(score)
    pairs = pairs.copy(); pairs["tfidf_score"]=scores
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT,index=False,encoding="utf-8-sig")
    print(pairs.groupby("label")["tfidf_score"].agg(["count","mean","min","max"]))

if __name__ == "__main__": main()

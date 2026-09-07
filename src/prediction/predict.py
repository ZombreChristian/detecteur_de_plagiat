"""Prédit la probabilité de similarité pour deux documents DOCX."""
from pathlib import Path
import joblib, numpy as np
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

MODEL_NAME="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
def read_docx(path):
    d=Document(path); parts=[p.text.strip() for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for r in t.rows: parts.append(" | ".join(c.text.strip() for c in r.cells))
    return " ".join(parts)

def predict(file_a,file_b):
    texts=[read_docx(Path(file_a)),read_docx(Path(file_b))]
    tf=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True); X=tf.fit_transform(texts); lexical=float(cosine_similarity(X[0],X[1])[0,0])
    model=SentenceTransformer(MODEL_NAME); E=model.encode(texts,normalize_embeddings=True); semantic=float(np.dot(E[0],E[1])); hybrid=.35*lexical+.65*semantic
    return {"tfidf_score":lexical,"semantic_score":semantic,"hybrid_score":hybrid,"decision":"SIMILAIRES" if hybrid>=0.70 else "A EXAMINER" if hybrid>=0.50 else "DIFFERENTS"}

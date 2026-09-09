"""API FastAPI du moteur de détection de plagiat et de doublons."""
from pathlib import Path
import tempfile, time, re
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from src.passages.passage_engine import analyze_document_pair

ROOT = Path(__file__).resolve().parents[1]
CORPORA = {"plagiarism": ROOT / "donnees" / "Rapport d'etude", "duplicate": ROOT / "donnees" / "TDR"}
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
THRESHOLDS = {"plagiarism": 0.55, "duplicate": 0.70}
app = FastAPI(title="Détecteur de plagiat et doublon", version="1.0.0")

def read_docx(path):
    doc = Document(path); parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows: parts.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n\n".join(parts)

def files_in_corpus(kind):
    folder = CORPORA[kind]
    return sorted(folder.glob("*.docx")) if folder.exists() else []

def score_documents(candidate, sources):
    texts = [candidate] + [read_docx(p) for p in sources]
    if len(texts) < 2: return []
    tf = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True).fit_transform(texts)
    lexical = cosine_similarity(tf[0:1], tf[1:]).ravel()
    emb = SentenceTransformer(MODEL_NAME).encode(texts, normalize_embeddings=True, show_progress_bar=False)
    semantic = np.matmul(emb[1:], emb[0]); rows = []
    for i, path in enumerate(sources):
        hybrid = float(.35 * lexical[i] + .65 * semantic[i])
        rows.append({"source":path.name,"tfidf_score":round(float(lexical[i]),4),"semantic_score":round(float(semantic[i]),4),"hybrid_score":round(hybrid,4),"_path":path})
    return sorted(rows, key=lambda x:x["hybrid_score"], reverse=True)

def decision(score, threshold):
    if score >= threshold: return "SIMILAIRE"
    if score >= threshold - .15: return "A EXAMINER"
    return "DIFFERENT"

def run_detection(file_path, kind):
    candidate = read_docx(file_path)
    if not candidate.strip(): raise HTTPException(400, "Le document ne contient aucun texte exploitable.")
    sources = files_in_corpus(kind)
    if not sources: raise HTTPException(500, f"Corpus introuvable ou vide: {CORPORA[kind]}")
    ranked = score_documents(candidate, sources); best = ranked[0]; passage_results = []
    for item in ranked[:5]:
        detail = analyze_document_pair(candidate, read_docx(item["_path"]), threshold=.50, top_k=5)
        passage_results.append({"source":item["source"],"document_score":item["hybrid_score"],**detail})
    threshold = THRESHOLDS[kind]
    return {"mode":kind,"decision":decision(best["hybrid_score"],threshold),"threshold":threshold,"tfidf_score":best["tfidf_score"],"semantic_score":best["semantic_score"],"hybrid_score":best["hybrid_score"],"novelty_score":round(1-best["hybrid_score"],4),"sources":[{k:v for k,v in x.items() if k != "_path"} for x in ranked[:10]],"matches":passage_results}

@app.get("/")
def root(): return {"service":"detecteur_de_plagiat","status":"ok"}
@app.get("/api/health")
def health(): return {"status":"ok","corpora":{k:len(files_in_corpus(k)) for k in CORPORA}}
@app.post("/api/detect/{kind}")
async def detect(kind: str, file: UploadFile = File(...)):
    if kind not in CORPORA: raise HTTPException(404,"Mode inconnu. Utilisez plagiarism ou duplicate.")
    if not file.filename.lower().endswith(".docx"): raise HTTPException(400,"Format accepté: DOCX.")
    start=time.perf_counter()
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/re.sub(r"[^\w.\- ]","_",file.filename); path.write_bytes(await file.read()); result=run_detection(path,kind)
    result["duration_ms"]=round((time.perf_counter()-start)*1000); return result
@app.post("/api/compare")
async def compare(file_a: UploadFile=File(...), file_b: UploadFile=File(...)):
    with tempfile.TemporaryDirectory() as d:
        a,b=Path(d)/"a.docx",Path(d)/"b.docx"; a.write_bytes(await file_a.read()); b.write_bytes(await file_b.read()); ranked=score_documents(read_docx(a),[b])[0]
        return {k:v for k,v in ranked.items() if k != "_path"}

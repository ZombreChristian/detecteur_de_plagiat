"""API minimale FastAPI pour comparer deux DOCX."""
from pathlib import Path
import tempfile
from fastapi import FastAPI, UploadFile, File
from src.prediction.predict import predict

app=FastAPI(title="Détecteur de plagiat et doublon")
@app.get("/")
def root(): return {"service":"detecteur_de_plagiat","status":"ok"}
@app.post("/api/compare")
async def compare(file_a:UploadFile=File(...),file_b:UploadFile=File(...)):
    with tempfile.TemporaryDirectory() as d:
        a=Path(d)/file_a.filename; b=Path(d)/file_b.filename
        a.write_bytes(await file_a.read()); b.write_bytes(await file_b.read())
        return predict(a,b)

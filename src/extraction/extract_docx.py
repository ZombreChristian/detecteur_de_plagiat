"""Extraction du texte des fichiers DOCX du corpus."""
from pathlib import Path
from docx import Document
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "donnees"
OUT = DATA / "documents_extraits.csv"

def extract_docx(path: Path) -> str:
    doc = Document(path)
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)

def main():
    rows = []
    for folder in [DATA / "TDR", DATA / "Rapport d'etude"]:
        for path in sorted(folder.glob("*.docx")):
            text = extract_docx(path)
            rows.append({"document_id": path.stem, "type_document": folder.name, "path": str(path.relative_to(ROOT)), "text": text, "nb_caracteres": len(text), "nb_mots": len(text.split())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"Documents extraits : {len(df)}")
    print(df.groupby("type_document").size())

if __name__ == "__main__":
    main()

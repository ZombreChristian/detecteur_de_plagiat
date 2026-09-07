"""Génère des paires positives/négatives sans fuite entre familles."""
from pathlib import Path
from itertools import combinations
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "donnees" / "dataset_metadata.csv"
OUT = ROOT / "donnees" / "pairs.csv"

def main():
    meta = pd.read_csv(META)
    rows = []
    for typ, g in meta.groupby("type_document"):
        records = g.to_dict("records")
        # Positifs : chaque document et sa source explicitement annotée.
        by_id = {r["document_id"]: r for r in records}
        for r in records:
            src = r.get("document_source")
            if pd.notna(src) and src and src in by_id:
                rows.append({"doc_a": src, "doc_b": r["document_id"], "type_document": typ, "label": 1, "relation": r["relation"]})
        # Négatifs : paires sans lien source, en privilégiant les mêmes secteurs.
        for a, b in combinations(records, 2):
            if a["document_id"] == b["document_id"] or a["document_id"] == b.get("document_source") or b["document_id"] == a.get("document_source"):
                continue
            same_sector = a["secteur"] == b["secteur"]
            if same_sector or (a["zone"] == b["zone"]):
                rows.append({"doc_a": a["document_id"], "doc_b": b["document_id"], "type_document": typ, "label": 0, "relation": "different_or_hard_negative"})
    out = pd.DataFrame(rows).drop_duplicates(subset=["doc_a", "doc_b"])
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(out["label"].value_counts())
    print(f"Total paires : {len(out)}")

if __name__ == "__main__":
    main()

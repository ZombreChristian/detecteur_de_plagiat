
"""
Baseline TF-IDF + similarité cosinus.

Entrées :
    donnees/documents_nettoyes.csv
    donnees/pairs.csv

Sortie :
    resultats/tfidf_predictions.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CHEMINS DU PROJET
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DOCS = ROOT / "donnees" / "documents_nettoyes.csv"
PAIRS = ROOT / "donnees" / "pairs.csv"
OUT = ROOT / "resultats" / "tfidf_predictions.csv"


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 60)
    print("BASELINE TF-IDF + SIMILARITÉ COSINUS")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Vérification des fichiers
    # --------------------------------------------------------

    if not DOCS.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DOCS}\n"
            "Exécute d'abord : "
            "python src/preprocessing/clean_text.py"
        )

    if not PAIRS.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {PAIRS}\n"
            "Exécute d'abord : "
            "python src/dataset/build_pairs.py"
        )

    # --------------------------------------------------------
    # 2. Chargement des documents nettoyés
    # --------------------------------------------------------

    print(f"\nDocuments utilisés : {DOCS}")

    docs = pd.read_csv(
        DOCS,
        encoding="utf-8-sig"
    ).set_index("document_id")

    print(f"Nombre de documents : {len(docs)}")

    # --------------------------------------------------------
    # 3. Chargement des paires
    # --------------------------------------------------------

    pairs = pd.read_csv(
        PAIRS,
        encoding="utf-8-sig"
    )

    print(f"Nombre de paires : {len(pairs)}")

    # --------------------------------------------------------
    # 4. Vérification de la colonne text
    # --------------------------------------------------------

    if "text" not in docs.columns:
        raise ValueError(
            "La colonne 'text' est absente de "
            "documents_nettoyes.csv."
        )

    # --------------------------------------------------------
    # 5. Transformation TF-IDF
    # --------------------------------------------------------

    print("\nCalcul des représentations TF-IDF...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(
        docs["text"].fillna("")
    )

    print(
        f"Matrice TF-IDF : "
        f"{X.shape[0]} documents × {X.shape[1]} termes"
    )

    # --------------------------------------------------------
    # 6. Correspondance document_id → position
    # --------------------------------------------------------

    ids = list(docs.index)

    positions = {
        document_id: i
        for i, document_id in enumerate(ids)
    }

    # --------------------------------------------------------
    # 7. Calcul de la similarité cosinus
    # --------------------------------------------------------

    print("\nCalcul des similarités...")

    scores = []

    for _, row in pairs.iterrows():

        a = row["doc_a"]
        b = row["doc_b"]

        if a not in positions:
            raise ValueError(
                f"Document absent de documents_nettoyes.csv : {a}"
            )

        if b not in positions:
            raise ValueError(
                f"Document absent de documents_nettoyes.csv : {b}"
            )

        score = float(
            cosine_similarity(
                X[positions[a]],
                X[positions[b]]
            )[0, 0]
        )

        scores.append(score)

    # --------------------------------------------------------
    # 8. Ajout des scores
    # --------------------------------------------------------

    pairs = pairs.copy()

    pairs["tfidf_score"] = scores

    # --------------------------------------------------------
    # 9. Création du dossier de sortie
    # --------------------------------------------------------

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 10. Sauvegarde
    # --------------------------------------------------------

    pairs.to_csv(
        OUT,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 11. Résumé des résultats
    # --------------------------------------------------------

    print("\nRésultats enregistrés dans :")
    print(OUT)

    print("\nStatistiques des scores TF-IDF :")

    print(
        pairs
        .groupby("label")["tfidf_score"]
        .agg(
            count="count",
            mean="mean",
            min="min",
            max="max"
        )
    )

    print("\nAperçu des résultats :")

    print(
        pairs[
            ["doc_a", "doc_b", "label", "tfidf_score"]
        ].head(10).to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("TF-IDF TERMINÉ")
    print("=" * 60)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    main()


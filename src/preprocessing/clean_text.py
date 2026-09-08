
"""
Nettoyage et normalisation du texte des documents.

Entrée :
    donnees/documents_extraits.csv

Sortie :
    donnees/documents_nettoyes.csv
"""

from pathlib import Path
import re
import unicodedata

import pandas as pd


# ============================================================
# CHEMINS DU PROJET
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "donnees"

INPUT_FILE = DATA / "documents_extraits.csv"
OUTPUT_FILE = DATA / "documents_nettoyes.csv"


# ============================================================
# FONCTION DE NETTOYAGE
# ============================================================

def clean_text(text: str) -> str:
    """
    Nettoie et normalise un texte français.

    Opérations réalisées :
    - conversion en chaîne de caractères ;
    - normalisation Unicode ;
    - remplacement des espaces insécables ;
    - suppression des retours à la ligne inutiles ;
    - réduction des espaces multiples ;
    - suppression des espaces en début et fin.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Normalisation Unicode
    text = unicodedata.normalize("NFKC", text)

    # Remplacement des espaces insécables
    text = text.replace("\u00a0", " ")

    # Remplacement des retours à la ligne et tabulations
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    # Suppression des espaces multiples
    text = re.sub(r"\s+", " ", text)

    # Suppression des espaces en début et fin
    text = text.strip()

    return text


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    print("=" * 60)
    print("NETTOYAGE DU CORPUS")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Vérification du fichier d'entrée
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}\n"
            "Exécute d'abord extract_docx.py."
        )

    print(f"\nFichier d'entrée : {INPUT_FILE}")

    # --------------------------------------------------------
    # 2. Lecture du fichier CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

    print(f"Documents chargés : {len(df)}")

    # Vérification de la colonne text
    if "text" not in df.columns:
        raise ValueError(
            "La colonne 'text' est absente du fichier "
            "documents_extraits.csv."
        )

    # --------------------------------------------------------
    # 3. Calcul du nombre de caractères avant nettoyage
    # --------------------------------------------------------

    df["nb_caracteres_avant"] = df["text"].fillna("").astype(str).str.len()

    # --------------------------------------------------------
    # 4. Nettoyage des textes
    # --------------------------------------------------------

    print("\nNettoyage des textes...")

    df["text"] = df["text"].apply(clean_text)

    # --------------------------------------------------------
    # 5. Calcul des statistiques après nettoyage
    # --------------------------------------------------------

    df["nb_caracteres_apres"] = df["text"].str.len()

    df["nb_mots"] = df["text"].apply(
        lambda x: len(x.split())
    )

    # --------------------------------------------------------
    # 6. Sauvegarde
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 7. Résultats
    # --------------------------------------------------------

    print("\nNettoyage terminé.")

    print(f"Fichier de sortie : {OUTPUT_FILE}")
    print(f"Nombre de documents : {len(df)}")

    print("\nStatistiques :")

    print(
        f"Caractères avant nettoyage : "
        f"{df['nb_caracteres_avant'].sum():,}"
    )

    print(
        f"Caractères après nettoyage : "
        f"{df['nb_caracteres_apres'].sum():,}"
    )

    print(
        f"Nombre moyen de mots/document : "
        f"{df['nb_mots'].mean():.2f}"
    )

    print("\nAperçu :")

    print(
        df[
            [
                "document_id",
                "type_document",
                "nb_caracteres_avant",
                "nb_caracteres_apres",
                "nb_mots"
            ]
        ].head(10).to_string(index=False)
    )

    print("\n" + "=" * 60)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    main()


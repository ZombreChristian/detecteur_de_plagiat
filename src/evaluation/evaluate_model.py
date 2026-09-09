"""
Évaluation du modèle de détection de similarité.

Le modèle est évalué sur un jeu de test séparé du jeu
utilisé pour son entraînement.

Entrée :
    resultats/semantic_predictions.csv
    modeles/similarity_classifier.joblib

Sorties :
    resultats/metrics/evaluation_model.txt
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# CHEMINS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "resultats" / "semantic_predictions.csv"
MODEL_PATH = ROOT / "modeles" / "similarity_classifier.joblib"
OUTPUT = ROOT / "resultats" / "metrics" / "evaluation_model.txt"


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

def load_data():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT}\n"
            "Exécute d'abord semantic_similarity.py."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Exécute d'abord train_model.py."
        )

    df = pd.read_csv(INPUT)

    required_columns = [
        "doc_a",
        "doc_b",
        "label",
        "tfidf_score",
        "semantic_score",
        "hybrid_score",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {INPUT} : {missing}"
        )

    return df


# ============================================================
# CRÉATION DU JEU DE TEST
# ============================================================

def create_test_split(df):
    """
    Reproduit le principe de séparation utilisé dans train_model.py.

    Le regroupement permet d'éviter autant que possible qu'une même
    famille de documents se retrouve dans l'entraînement et le test.
    """

    groups = (
        df["doc_a"]
        .astype(str)
        .str.extract(r"(\D+-?\d+)")[0]
        .fillna(df["doc_a"])
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            df["label"],
            groups
        )
    )

    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


# ============================================================
# ÉVALUATION
# ============================================================

def evaluate():

    print("=" * 70)
    print("ÉVALUATION DU MODÈLE DE DÉTECTION DE SIMILARITÉ")
    print("=" * 70)

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    df = load_data()

    print(f"\nNombre total de paires : {len(df)}")

    # --------------------------------------------------------
    # Séparation train/test
    # --------------------------------------------------------

    train_df, test_df = create_test_split(df)

    print(f"Nombre de paires entraînement : {len(train_df)}")
    print(f"Nombre de paires test         : {len(test_df)}")

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    features = [
        "tfidf_score",
        "semantic_score",
        "hybrid_score"
    ]

    X_test = test_df[features]
    y_test = test_df["label"].astype(int)

    # --------------------------------------------------------
    # Chargement du modèle
    # --------------------------------------------------------

    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------------
    # Prédictions
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # Métriques
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    try:
        roc_auc = roc_auc_score(
            y_test,
            y_proba
        )
    except ValueError:
        roc_auc = float("nan")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )

    # --------------------------------------------------------
    # Affichage
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("RÉSULTATS")
    print("-" * 70)

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nMatrice de confusion :")
    print(cm)

    print("\nRapport de classification :")
    print(report)

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_text = f"""
ÉVALUATION DU MODÈLE DE DÉTECTION DE SIMILARITÉ
================================================

Données
-------
Nombre total de paires       : {len(df)}
Nombre de paires entraînement: {len(train_df)}
Nombre de paires test        : {len(test_df)}

Caractéristiques utilisées
--------------------------
- tfidf_score
- semantic_score
- hybrid_score

Métriques
---------
Accuracy  : {accuracy:.4f}
Precision : {precision:.4f}
Recall    : {recall:.4f}
F1-score  : {f1:.4f}
ROC-AUC   : {roc_auc:.4f}

Matrice de confusion
--------------------
{cm}

Rapport de classification
-------------------------
{report}
"""

    OUTPUT.write_text(
        result_text,
        encoding="utf-8"
    )

    print(f"\nÉvaluation sauvegardée dans :")
    print(OUTPUT)

    print("\n" + "=" * 70)
    print("ÉVALUATION TERMINÉE")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()

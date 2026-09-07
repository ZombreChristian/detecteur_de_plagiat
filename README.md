# Détecteur de plagiat et de doublon d'études

Prototype expérimental basé sur le cahier des charges du projet.

## Objectif
- détecter les quasi-doublons de TDR;
- détecter les copies et paraphrases dans les rapports;
- fournir un score lexical, sémantique et hybride;
- préparer une API d'intégration.

## Pipeline
1. Extraction DOCX
2. Nettoyage
3. Construction des paires annotées
4. Baseline TF-IDF
5. Embeddings Sentence Transformer
6. Score hybride
7. Classification Logistic Regression
8. Évaluation
9. API FastAPI

## Installation
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/WSL
source .venv/bin/activate
pip install -r requirements.txt
```

## Exécution
Depuis la racine du dépôt :
```bash
python src/extraction/extract_docx.py
python src/dataset/build_pairs.py
python src/embeddings/generate_embeddings.py
python src/similarity/semantic_similarity.py
python src/training/train_model.py
```

API :
```bash
uvicorn api.app:app --reload
```

> Les modèles et résultats générés localement ne sont pas versionnés par Git. Le corpus de test est synthétique et sert à valider le pipeline avant intégration de données réelles.

# DOCSEC — Détecteur de plagiat et de doublon d'études

Prototype de détection documentaire conforme au cahier des charges : détection de quasi-doublons de TDR, détection de copies/paraphrases dans les rapports, explicabilité par sources et passages, historique des analyses et export des résultats.

## Architecture

```text
DOCX
  ↓
Extraction et nettoyage
  ↓
TF-IDF + Sentence Transformer multilingue
  ↓
Score hybride lexical/sémantique
  ↓
Pré-sélection des sources
  ↓
Localisation des passages similaires
  ↓
FastAPI
  ↓
Django + PostgreSQL
```

## Corpus

- `donnees/TDR/` : TDR utilisés pour la détection de doublons.
- `donnees/Rapport d'etude/` : rapports utilisés pour la détection de plagiat.
- `donnees/dataset_metadata.csv` : métadonnées et relations attendues du corpus synthétique.

Le corpus est synthétique et sert à valider le pipeline avant intégration de données réelles.

## Installation

Depuis la racine :

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/WSL
source .venv/bin/activate
pip install -r requirements.txt
```

## PostgreSQL

Le projet utilise PostgreSQL et non SQLite.

```bash
docker compose up -d postgres
```

Créer ensuite `frontend/.env` à partir de `frontend/.env.example`.

## Base Django

```bash
cd frontend
python manage.py migrate
python manage.py createsuperuser
```

## Lancer l'application

Terminal 1, depuis la racine :

```bash
uvicorn api.app:app --reload --port 8000
```

Terminal 2 :

```bash
cd frontend
python manage.py runserver 127.0.0.1:8001
```

Interfaces :

- Django : `http://127.0.0.1:8001/`
- Administration : `http://127.0.0.1:8001/admin/`
- Documentation API : `http://127.0.0.1:8000/docs`
- Santé API : `http://127.0.0.1:8000/api/health`

## Fonctionnalités

- Analyse plagiat des rapports.
- Analyse doublon des TDR.
- TF-IDF pour le recouvrement lexical.
- Sentence Transformer multilingue pour la similarité sémantique.
- Score hybride : 35 % lexical + 65 % sémantique.
- Classement des sources les plus proches.
- Localisation des passages similaires.
- Score de nouveauté.
- Seuils distincts pour plagiat et doublon.
- Historique PostgreSQL.
- Administration Django.
- Export PDF et Excel.
- Whitelist des passages administratifs récurrents via Django Admin.

## API

`POST /api/detect/plagiarism` avec un champ multipart `file`.

`POST /api/detect/duplicate` avec un champ multipart `file`.

`POST /api/compare` permet de comparer directement deux DOCX.

## Important pour les essais

Le moteur ne doit pas comparer le document envoyé avec lui-même. Les sources sont recherchées dans le corpus correspondant au mode choisi. Les seuils sont des valeurs initiales et doivent être calibrés à partir des résultats d'évaluation sur des données annotées.

Les modèles et résultats générés localement ne sont pas versionnés par Git.

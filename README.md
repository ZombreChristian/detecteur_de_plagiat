# Détecteur de plagiat et de doublon d'études

Plateforme de détection documentaire conforme au cahier des charges : doublons de TDR, plagiat littéral/paraphrasé dans les rapports, explicabilité par sources et passages, journalisation et stockage PostgreSQL.

## Architecture
- **Moteur ML** : TF-IDF + Sentence Transformer multilingue + score hybride.
- **Localisation** : comparaison passage par passage.
- **API** : FastAPI (`api/app.py`).
- **Interface** : Django (`frontend/`).
- **Base** : PostgreSQL.
- **Administration** : Django Admin pour documents, analyses et whitelist.

## Arborescence principale
```text
api/                    API FastAPI
donnees/TDR/            corpus TDR
donnees/Rapport d'etude/ corpus rapports
src/passages/           localisation des passages
src/prediction/         détection d'une paire
src/training/           apprentissage/évaluation
frontend/               application Django
  detector/             modèles, vues, migrations
  templates/            interface web
  static/               CSS
docker-compose.yml      PostgreSQL
```

## Installation
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/WSL
source .venv/bin/activate
pip install -r requirements.txt
```

### PostgreSQL avec Docker
```bash
docker compose up -d postgres
```
Le service crée la base `detecteur_plagiat` sur `localhost:5432`.

Copiez `frontend/.env.example` vers `frontend/.env` et adaptez le mot de passe si nécessaire. Les variables peuvent aussi être définies directement dans l'environnement système.

## Initialisation Django
```bash
cd frontend
python manage.py migrate
python manage.py createsuperuser
```

## Démarrage
Ouvrir deux terminaux.

Terminal 1 — API :
```bash
uvicorn api.app:app --reload --port 8000
```

Terminal 2 — Django :
```bash
cd frontend
python manage.py runserver 127.0.0.1:8001
```

Interface : `http://127.0.0.1:8001/`
Administration : `http://127.0.0.1:8001/admin/`
Documentation API : `http://127.0.0.1:8000/docs`

## API
- `GET /api/health` : état du service et taille des corpus.
- `POST /api/detect/plagiarism` : analyse d'un rapport DOCX.
- `POST /api/detect/duplicate` : analyse d'un TDR DOCX.
- `POST /api/compare` : comparaison directe de deux DOCX.

Le champ multipart attendu pour les détections est `file`.

## Pipeline scientifique
1. Extraction du document.
2. Nettoyage/normalisation.
3. Segmentation en passages.
4. Préfiltrage lexical TF-IDF.
5. Encodage sémantique multilingue.
6. Fusion TF-IDF (35 %) + sémantique (65 %).
7. Classement des documents sources.
8. Localisation des passages similaires.
9. Décision selon un seuil configurable par moteur.
10. Calcul du score de nouveauté et conservation du résultat dans PostgreSQL.

Les seuils actuels sont des seuils initiaux à calibrer sur le corpus réel : `0.55` pour les rapports et `0.70` pour les TDR. Ils ne constituent pas une preuve juridique automatique.

## Données
Le corpus synthétique permet de valider le pipeline. Pour une mise en production, les données réelles doivent être importées avec leurs métadonnées : titre, objet, portée géographique, secteur, résultats attendus, entité commanditaire, année, budget, statut et document source.

## Important
Les modèles lourds et les résultats générés localement restent exclus du versionnage. Les données administratives réelles doivent être hébergées selon les exigences de souveraineté, confidentialité, droits d'accès et conservation prévues par le projet.

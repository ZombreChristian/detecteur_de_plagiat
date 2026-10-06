# TDRDOC-SCAN — Détecteur de plagiat et de doublon d'études

Prototype de détection documentaire : détection de quasi-doublons de TDR, détection de copies/paraphrases dans les rapports, explicabilité par sources et passages, historique des analyses et export des résultats.

## Architecture de similarité

```text
DOCX de référence
  ↓
Extraction
  ↓
Nettoyage / prétraitement
  ↓
Embedding Sentence Transformer
  ↓
PostgreSQL
  ├── métadonnées
  ├── texte extrait
  ├── texte nettoyé
  └── embedding
```

Pour chaque fichier uploadé par un utilisateur :

```text
DOCX candidat
  ↓
Extraction
  ↓
Nettoyage / prétraitement
  ├───────────────┐
  ▼               ▼
TF-IDF       Embedding
  │               │
  ▼               ▼
Score lexical  Similarité cosinus
  │               │
  └───────┬───────┘
          ▼
Score hybride
30 % lexical + 70 % sémantique
          ↓
Classement des sources
          ↓
Analyse détaillée des passages des meilleures sources
          ↓
Résultat
```

Le fichier uploadé est un **candidat de comparaison uniquement**. Il n'est pas ajouté automatiquement au registre de référence. Seuls les résultats d'analyse sont conservés dans l'historique.

## Corpus

- `donnees/TDR/` : TDR de référence pour la détection de doublons.
- `donnees/Rapport d'etude/` : rapports de référence pour la détection de plagiat.
- `donnees/documents_extraits.csv` et `donnees/documents_nettoyes.csv` restent utiles pour les traitements offline existants.

## Modèle

Modèle sémantique :

NaNsentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Lors de la première utilisation, il est téléchargé depuis Hugging Face puis sauvegardé localement dans :

NaNmodeles/paraphrase-multilingual-MiniLM-L12-v2/`

Les exécutions suivantes chargent uniquement cette copie locale. Les embeddings des documents de référence sont quant à eux stockés dans PostgreSQL.

Un `HF_TOKEN` est facultatif, mais permet un téléchargement authentifié depuis le Hub.

## PostgreSQL

Le projet utilise PostgreSQL.

```bash
docker compose up -d postgres
```

Créer ensuite `frontend/.env` à partir de `frontend/.env.example`.

## Première exécution

Depuis la racine du projet :

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/WSL
source .venv/bin/activate

pip install -r requirements.txt

cd frontend
python manage.py migrate
python manage.py createsuperuser
cd ..
```

Puis lancer FastAPI :

```bash
uvicorn api.app:app --reload --port 8000
```

Au premier démarrage, FastAPI :

1. charge/télécharge le modèle puis le sauvegarde localement ;
2. vérifie/applique les migrations Django ;
3. extrait et nettoie le corpus ;
4. calcule les embeddings des documents de référence ;
5. stocke ces embeddings dans PostgreSQL ;
6. prépare en mémoire l'index TF-IDF utilisé pendant les requêtes.

Les démarrages suivants ne recalculent pas les embeddings déjà indexés.

Pour forcer une réindexation du corpus :

```bash
cd frontend
python manage.py prepare_reference_index --force
```

Sans `--force`, la commande ne recalcule que les documents absents, modifiés ou non indexés.

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
- Documentation API : `http://127.0.0.1:8000/docs`
- Santé API : `http://127.0.0.1:8000/api/health`

## Calcul des scores

Score hybride = 30 % lexical + 70 % sémantique.

- **Score lexical** : cosine similarity entre les représentations TF-IDF du candidat et des références.
- **Score sémantique** : cosine similarity entre les embeddings produits par Sentence Transformer.
- **Score hybride** : fusion 30/70.
- Les passages similaires sont recherchés uniquement sur les meilleures sources afin de limiter le temps de calcul.

Les seuils actuels sont :

- doublon TDR : `0,70`
- plagiat rapport : `0,55`

Ils restent des valeurs initiales et doivent être calibrés avec des données annotées.

## API

- `POST /api/detect/duplicate`
- `POST /api/detect/plagiarism`
- `POST /api/compare`

## Base Django

NaNStudyDocument` stocke les documents de référence et leurs données préparées : texte extrait, texte nettoyé, embedding, hash du contenu et date d'indexation.

NaNAnalysis` stocke l'historique des contrôles : scores lexical/sémantique/hybride, décision, sources et détails des passages.

## Branche de développement

Toutes les modifications de cette évolution sont réalisées uniquement sur la branche `feature/testsimilarity`.
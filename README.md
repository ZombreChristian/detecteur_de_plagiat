# TDRDOC-SCAN — Détecteur de plagiat et de doublon d'études

TDRDOC-SCAN est une application de contrôle documentaire destinée à comparer les termes de référence (TDR) et à détecter les ressemblances entre rapports d'études. Elle s'appuie sur une interface Django, une API FastAPI, PostgreSQL, une recherche lexicale et sémantique et un arbitrage par modèle de langage pour l'analyse des TDR.

> **Branche de développement suivie :** `feature/ollama-optimization-admin-testv1`.
> Les modifications décrites dans cette branche ne doivent pas être fusionnées dans `main` sans autorisation explicite.

## Fonctionnalités

### Vérification des TDR (mode « Doublon »)

1. Le texte du document DOCX ou PDF est extrait et préparé.
2. Le moteur lexical et sémantique classe les documents de référence indexés dans PostgreSQL.
3. Les références candidates sont transmises au moteur RAG et au modèle configuré.
4. L'analyse compare le contenu métier : objet, problématique, objectifs, activités, zone géographique, population cible, période, méthodologie et livrables, lorsque ces informations sont disponibles.
5. L'application retourne un verdict expliqué, les références retenues, les points communs, les différences et une recommandation.

Les scores TF-IDF, sémantiques et hybrides servent à **retrouver et classer les références candidates**. Ils ne doivent pas, à eux seuls, déterminer le verdict métier du TDR.

Les niveaux de conclusion utilisés par le moteur RAG indépendant sont :
- **Correspondance forte** : le TDR reprend la même étude ou une étude quasi identique.
- **Correspondance partielle** : des dimensions métier importantes se recoupent, mais des différences significatives subsistent.
- **Aucune étude pertinente** : aucune référence examinée ne présente de correspondance métier suffisante.

Une zone géographique différente n'exclut pas automatiquement une correspondance partielle si les objectifs et la démarche se recoupent réellement.

Si le fournisseur LLM est indisponible ou renvoie une réponse inexploitable, l'analyse doit signaler l'indisponibilité au lieu de fabriquer un verdict à partir des seuls scores de recherche.

### Analyse des rapports (mode « Plagiat »)

Le parcours des rapports conserve la recherche lexicale et sémantique, l'analyse des passages et le mécanisme de décision par seuil du mode plagiat. Les poids par défaut de la recherche hybride sont de 30 % pour le lexical et de 70 % pour le sémantique.

### Registre et index de référence

Les documents de référence sont préparés et indexés séparément. Un document candidat soumis à l'analyse ne doit pas être ajouté automatiquement au registre des références.

## Architecture technique

- **Interface web :** Django, dans `frontend/`.
- **API :** FastAPI, dans `api/app.py`.
- **Application Django :** `frontend/detector/`.
- **Moteur RAG TDR indépendant :** `src/tdr_rag/independent_engine.py`.
- **Recherche documentaire :** TF-IDF, embeddings multilingues et similarité cosinus.
- **Base de données :** PostgreSQL.
- **Modèle d'embeddings par défaut :** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- **LLM TDR :** fournisseur compatible OpenAI ou Ollama, selon `TDR_LLM_PROVIDER` et la configuration de l'environnement.

## Prérequis

- Python 3.10 ou version compatible avec les dépendances du projet ;
- PostgreSQL ;
- Git ;
- Ollama si le fournisseur local est utilisé ;
- un modèle Qwen disponible auprès du fournisseur configuré.

## Installation sous Windows

Depuis PowerShell, se placer à la racine du dépôt :

```powershell
cd C:\Users\Leslie\Desktop\Plagiat\detecteur_de_plagiat
```

### 1. Créer ou activer l'environnement virtuel

Pour créer l'environnement :

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Pour réactiver un environnement existant :

```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Installer les dépendances

```powershell
pip install -r requirements.txt
```

### 3. Configurer l'environnement

Créer le fichier local `frontend/.env` à partir du modèle :

```powershell
Copy-Item frontend/.env.example frontend/.env
```

Adapter ensuite les paramètres PostgreSQL, Django et les paramètres des services. **Ne jamais committer ni publier le vrai fichier `.env`**, car il peut contenir des mots de passe, clés ou autres secrets. Le fichier `.env.example` ne doit contenir que des valeurs fictives ou non sensibles.

Paramètres principaux présents dans `frontend/.env.example` :

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
OLLAMA_TIMEOUT_SECONDS=300

SIMILARITY_LEXICAL_WEIGHT=0.30
SIMILARITY_SEMANTIC_WEIGHT=0.70
SIMILARITY_REFERENCE_TOP_K=10
SIMILARITY_PASSAGE_TOP_K=3
```

Pour le moteur RAG indépendant, le fournisseur peut être sélectionné avec `TDR_LLM_PROVIDER` :
- `ollama` : utilise le service Ollama local et le modèle défini par `TDR_RAG_MODEL` (ou `OLLAMA_MODEL`) ;
- `openai_compatible` : utilise un serveur compatible avec l'API OpenAI, configuré avec `TDR_LLM_BASE_URL`, `TDR_LLM_API_KEY` et `TDR_LLM_MODEL`.

Ne placez jamais de vraie clé API dans `.env.example`, dans le code source ou dans le README. Le fournisseur et le modèle effectivement utilisés doivent être configurés explicitement dans le fichier local `.env`.

### 4. Préparer Ollama (uniquement si le fournisseur local est utilisé)

Installer Ollama depuis https://ollama.com/, puis :

```powershell
ollama pull qwen3:8b
ollama list
```

Vérifier que le service Ollama est démarré et que le nom du modèle correspond à la configuration locale.

## Préparer l'index documentaire

Depuis la racine du dépôt :

```powershell
cd frontend
python manage.py migrate
python manage.py prepare_reference_index
```

Cette commande prépare les documents de référence attendus par le projet, extrait leur contenu et calcule les données nécessaires à la recherche. Ne relancez pas une réindexation forcée à chaque analyse.

Pour forcer une réindexation complète, uniquement si nécessaire :

```powershell
python manage.py prepare_reference_index --force
```

## Démarrer les services

Lancer les services dans des terminaux PowerShell distincts, avec l'environnement virtuel activé.

### Terminal 1 — API FastAPI

Depuis la racine du dépôt :

```powershell
uvicorn api.app:app --reload --port 8000
```

### Terminal 2 — Interface Django

```powershell
cd frontend
python manage.py runserver 127.0.0.1:8001
```

Adresses locales :
- Interface Django : http://127.0.0.1:8001/
- Santé de l'API : http://127.0.0.1:8000/api/health
- Documentation FastAPI : http://127.0.0.1:8000/docs

## Configuration VS Code

Le dépôt contient `.vscode/settings.json` pour aider Pylance à résoudre les imports Django du dossier `frontend/`, notamment `detector.models`.

Après récupération des changements Git, si Pylance affiche encore une erreur d'import :
1. vérifier que VS Code a ouvert la racine du dépôt ;
2. sélectionner l'interpréteur Python de l'environnement virtuel `venv` ;
3. exécuter **Python: Restart Language Server** depuis la palette de commandes.

Cette configuration aide l'analyse statique de VS Code ; elle ne remplace pas l'installation des dépendances ni la configuration Django.

## Dépannage

**Le fournisseur LLM ne répond pas**
- Pour Ollama, vérifier que le service est démarré avec `ollama list`.
- Vérifier `TDR_LLM_PROVIDER`, l'URL du fournisseur et le nom du modèle dans le fichier local `.env`.
- Pour un serveur compatible OpenAI, vérifier l'URL, la clé et le nom du modèle.
- Consulter le message d'erreur renvoyé par l'API.

**Aucune référence n'est disponible**
- Vérifier que PostgreSQL est démarré et que les paramètres de connexion sont corrects.
- Vérifier la présence des documents de référence dans les emplacements prévus.
- Exécuter les migrations puis `python manage.py prepare_reference_index` depuis `frontend/`.

**Pylance ne trouve pas `detector.models`**
- Ouvrir la racine du dépôt dans VS Code.
- Vérifier l'interpréteur Python sélectionné.
- Vérifier les chemins `python.analysis.extraPaths` dans `.vscode/settings.json`.
- Redémarrer le serveur de langage Python.

## Sécurité

- Ne jamais committer `frontend/.env`, les mots de passe, clés API ou jetons d'accès.
- Si un secret a déjà été publié dans Git, le révoquer ou le renouveler. Le retirer dans un nouveau commit ne l'efface pas de l'historique Git.
- Utiliser une valeur sécurisée pour `DJANGO_SECRET_KEY` hors environnement de développement.
- Conserver uniquement des valeurs fictives dans `.env.example`.

## Branche de développement

La branche de travail suivie pour cette évolution est `feature/ollama-optimization-admin-testv1`. Ne pas pousser les changements sur `main` ni fusionner vers une autre branche sans validation explicite.

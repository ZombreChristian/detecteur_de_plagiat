# TDRDOC-SCAN — Détecteur de plagiat et de doublon d'études

TDRDOC-SCAN compare les termes de référence (TDR) soumis par les utilisateurs à un corpus de références indexé et détecte les ressemblances entre rapports d'études. L'application comprend une interface Django, une API FastAPI, PostgreSQL, un moteur lexical/sémantique et un arbitrage RAG local par Qwen3 via Ollama pour les TDR.

> **Branche de développement :** `feature/testsimilarity`. Les changements décrits ici ne doivent pas être fusionnés dans `main` sans autorisation explicite.

## Fonctionnement

### Analyse des TDR : recherche de références + arbitrage RAG

1. Le texte du TDR déposé est extrait et nettoyé.
2. Le moteur lexical et sémantique recherche les références les plus pertinentes dans l'index PostgreSQL.
3. Les références récupérées et le TDR candidat sont transmis à **Qwen3 via Ollama local**.
4. Qwen3 compare notamment l'objet, les activités, la zone géographique, les objectifs, les résultats attendus et, lorsque l'information est disponible, la population cible, la période et la méthodologie.
5. L'application présente le cas métier retenu, son explication, les points communs, les différences, les éléments probants et les informations manquantes.

Les scores lexicaux, sémantiques et hybrides servent à **retrouver les références**. Ils ne déterminent pas à eux seuls le verdict final d'un TDR.

Les quatre cas métier sont :
- **Cas 1 — Correspondance forte** : une référence couvre essentiellement la même étude.
- **Cas 2 — Correspondance partielle** : plusieurs dimensions convergent, mais des différences ou incertitudes importantes subsistent.
- **Cas 3 — Même objectif, zone différente** : l'objet ou les objectifs sont comparables, mais la zone géographique est explicitement différente.
- **Cas 4 — Aucune étude pertinente** : aucune des références récupérées n'est suffisamment pertinente.

Le modèle doit justifier sa décision à partir du TDR et des références fournies. Si Ollama est inaccessible, si le modèle manque ou si sa réponse est inexploitable, l'API renvoie une erreur explicite : elle ne remplace pas silencieusement l'arbitrage RAG par une décision basée uniquement sur les scores.

### Analyse des rapports d'études

Le mode rapports conserve le calcul de similarité lexicale et sémantique, l'analyse des passages et le seuil de décision du mode plagiat. Les poids par défaut du score hybride sont de 30 % pour le lexical et de 70 % pour le sémantique.

### Index de référence

Les documents de référence sont préparés et indexés séparément. Un document candidat envoyé pour analyse **n'est pas ajouté automatiquement** au corpus de référence.

## Architecture

- **Interface utilisateur :** Django dans `frontend/`
- **API de détection :** FastAPI dans `api/app.py`
- **Base de données :** PostgreSQL
- **Extraction :** documents DOCX et PDF pris en charge par l'API
- **Recherche de références :** TF-IDF, embeddings multilingues et similarité cosinus
- **Arbitrage des TDR :** Qwen3 local via Ollama et contexte de références récupérées (RAG)
- **Modèle d'embeddings par défaut :** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

## Prérequis

Installer les outils suivants :

- Python 3.11 recommandé ;
- PostgreSQL ;
- Git ;
- Ollama pour l'analyse RAG des TDR ;
- le modèle local Qwen3 choisi pour le projet, par défaut `qwen3:8b).

Vérifier que la machine dispose des ressources suffisantes pour exécuter le modèle local. La première utilisation du modèle d'embeddings peut nécessiter un téléchargement depuis Hugging Face.

## Installation sous Windows

Ouvrir PowerShell et se placer dans le dossier du dépôt :

```powershell
cd C:\Users\Leslie\Desktop\Plagiat\detecteur_de_plagiat
```

### 1. Créer ou activer l'environnement virtuel

Si l'environnement virtuel n'existe pas encore :

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

S'il existe déjà :

```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Installer les dépendances Python

Depuis la racine du dépôt :

```powershell
pip install -r requirements.txt
```

### 3. Configurer les variables d'environnement

Créer `frontend/.env` en copiant le modèle :

```powershell
Copy-Item frontend/.env.example frontend/.env
```

Ouvrir ensuite `frontend/.env` et adapter les paramètres PostgreSQL, Django et, si nécessaire, FastAPI et les e-mails. **Ne jamais publier le vrai fichier `.env` ni ses mots de passe.** Le fichier `.env.example` contient des valeurs d'exemple à remplacer.

Les paramètres RAG sont documentés dans `frontend/.env.example` :

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
OLLAMA_TIMEOUT_SECONDS=180
TDR_RAG_REFERENCE_TOP_K=3
TDR_RAG_CANDIDATE_MAX_CHARS=18000
TDR_RAG_REFERENCE_MAX_CHARS=7000
```

Ces variables contrôlent l'adresse d'Ollama, le modèle, le délai maximal et la quantité de texte transmise au modèle. Il n'est pas nécessaire de copier le contenu d'un fichier `.env` privé dans le dépôt.

### 4. Installer et vérifier Ollama pour l'analyse des TDR

Installer Ollama depuis le site officiel : https://ollama.com/

Puis, dans un terminal :

```powershell
ollama pull qwen3:8b
ollama list
```

Vérifier qu'Ollama est démarré et que le modèle configuré dans `OLLAMA_MODEL` apparaît dans la liste. Le service doit être accessible depuis la machine où FastAPI s'exécute. Si le modèle est absent ou si Ollama ne répond pas, l'arbitrage RAG des TDR échouera explicitement.

## Préparer la base de référence

Depuis la racine du dépôt :

```powershell
cd frontend
python manage.py migrate
python manage.py prepare_reference_index
```

La commande d'indexation prépare les documents de référence présents dans les emplacements configurés par le projet. Elle extrait et nettoie leur texte, calcule les embeddings et enregistre les informations d'indexation dans la base.

Si le contenu d'une référence n'a pas changé et qu'il est déjà indexé, son embedding peut être réutilisé. Pour forcer une réindexation complète, uniquement si nécessaire :

```powershell
python manage.py prepare_reference_index --force
```

Ne pas lancer cette commande avant chaque analyse. Vérifier que les documents de référence sont placés dans les dossiers attendus par la configuration du projet et que les deux corpus (TDR et rapports) sont correctement alimentés.

## Démarrer l'application

Lancer les services dans des terminaux séparés.

### Terminal 1 — API FastAPI

Depuis la racine du dépôt, avec l'environnement virtuel activé :

```powershell
uvicorn api.app:app --reload --port 8000
```

### Terminal 2 — Interface Django

Dans un deuxième terminal :

```powershell
cd C:\Users\Leslie\Desktop\Plagiat\detecteur_de_plagiat
.\venv\Scripts\Activate.ps1
cd frontend
python manage.py runserver 127.0.0.1:8001
```

Si l'environnement virtuel est déjà activé, ne l'activez pas une deuxième fois.

Accès locaux :
- Interface Django : http://127.0.0.1:8001/
- État de l'API : http://127.0.0.1:8000/api/health
- Documentation interactive FastAPI : http://127.0.0.1:8000/docs

## Paramètres de similarité

Les paramètres suivants sont définis dans `frontend/.env.example` et peuvent être adaptés dans `frontend/.env` :

- `SIMILARITY_LEXICAL_WEIGHT=0.30` : poids du score lexical ;
- `SIMILARITY_SEMANTIC_WEIGHT=0.70` : poids du score sémantique ;
- `SIMILARITY_REFERENCE_TOP_K=10` : nombre de références considérées par la recherche générale ;
- `SIMILARITY_PASSAGE_TOP_K=3` : nombre de sources utilisées pour l'analyse détaillée des passages ;
- `TDR_RAG_REFERENCE_TOP_K=3` : nombre maximal de références envoyées à Qwen3 pour l'arbitrage RAG.

La somme des poids lexical et sémantique doit être égale à 1. Les seuils de similarité restent utilisés pour le mode rapports. Pour les TDR, le verdict métier provient de l'arbitrage Qwen3 à partir des références récupérées, pas d'un seuil numérique fixe.

## Dépannage

**Ollama est inaccessible ou le modèle est introuvable**
- Vérifier que l'application Ollama est démarrée.
- Exécuter `ollama list` et confirmer le nom exact du modèle.
- Vérifier `OLLAMA_BASE_URL` et `OLLAMA_MODEL` dans `frontend/.env`.
- Consulter le message d'erreur retourné par l'API. Une indisponibilité de Qwen3 empêche de conclure l'analyse RAG du TDR.

**Aucune référence n'est trouvée**
- Vérifier que PostgreSQL est démarré et que les paramètres de connexion sont corrects.
- Vérifier que les documents de référence attendus sont présents.
- Exécuter les migrations Django, puis `python manage.py prepare_reference_index` depuis `frontend/`.

**Le modèle d'embeddings se télécharge au premier lancement**
- Cela peut être normal si le modèle n'est pas encore présent dans le répertoire local configuré.
- Vérifier la connexion Internet et l'espace disque. Les lancements suivants peuvent réutiliser la copie locale.

**L'API ne démarre pas**
- Vérifier que l'environnement virtuel est activé et que les dépendances sont installées.
- Vérifier que le terminal se trouve à la racine du dépôt avant de lancer `uvicorn api.app:app --reload --port 8000`.

## Sécurité

- Ne pas committer `frontend/.env`, mots de passe, clés secrètes ou jetons d'accès.
- Utiliser des valeurs sécurisées pour `DJANGO_SECRET_KEY` en dehors du développement local.
- Les valeurs de `.env.example` sont des exemples et doivent être adaptées à l'environnement utilisé.

## Branche de développement

Le travail de cette évolution se fait sur `feature/testsimilarity`. Ne pas modifier `main`, `suite_developpement` ou `feature/fusion` dans le cadre de cette évolution sans autorisation explicite.

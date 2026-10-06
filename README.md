# TDRDOC-SCAN — Détecteur de plagiat et de doublon d'études

Application de détection de doublons de TDR et de plagiat de rapports d'études. Le moteur combine une similarité lexicale et une similarité sémantique.

## 1. Principe général

Le projet fonctionne en deux phases distinctes.

### Phase A — Préparation du corpus de référence

Cette phase est réalisée avant les analyses utilisateurs.

```text
Documents DOCX de référence
        ↓
Extraction du texte
        ↓
Nettoyage / prétraitement
        ↓
Calcul des embeddings
        ↓
Stockage dans PostgreSQL
        ↓
Index prêt pour les analyses
```

Les embeddings des documents de référence sont donc calculés une fois et réutilisés. Ils ne sont pas recalculés à chaque upload.

### Phase B — Analyse d'un fichier uploadé

```text
Fichier DOCX uploadé
        ↓
Extraction
        ↓
Nettoyage
        ├──────────────────┐
        ↓                  ↓
     TF-IDF             Embedding
        ↓                  ↓
Score lexical       Similarité cosinus
        └─────────┬────────┘
                  ↓
       Score hybride 30/70
                  ↓
       Classement des références
                  ↓
     Analyse détaillée des passages
                  ↓
               Résultat
```

**Important :** le fichier uploadé est uniquement un candidat de comparaison. Il n'est pas ajouté automatiquement à la base des documents de référence.

## 2. Architecture des données

PostgreSQL contient notamment les informations préparées pour chaque document de référence :

- métadonnées du document ;
- texte extrait ;
- texte nettoyé ;
- embedding ;
- modèle utilisé pour l'embedding ;
- hash du contenu ;
- date d'indexation.

Le modèle de similarité est sauvegardé localement afin de ne pas être téléchargé à chaque démarrage.

## 3. Prérequis

Installer :

- Python 3.11 recommandé ;
- PostgreSQL ;
- Git ;
- les dépendances Python du projet.

Le projet peut également utiliser Docker pour PostgreSQL si le fichier `docker-compose.yml` est configuré pour cela.

## 4. Installation sous Windows

Depuis PowerShell, placer le terminal dans la racine du projet :

```powershell
cd C:\Users\Leslie\Desktop\Plagiat\detecteur_de_plagiat
```

### 4.1 Créer l'environnement virtuel

```powershell
python -m venv venv
venv\Scripts\activate
```

Si l'environnement existe déjà, il suffit de l'activer :

```powershell
venv\Scripts\activate
```

### 4.2 Installer les dépendances

Depuis la racine du projet :

```powershell
pip install -r requirements.txt
```

## 5. Configurer PostgreSQL

Créer le fichier `frontend/.env` à partir de `frontend/.env.example`.

Les paramètres PostgreSQL doivent correspondre à la base utilisée par Django.

Exemple :

```text
POSTGRES_DB=detecteur_plagiat
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
```

Si PostgreSQL est lancé avec Docker et que le projet fournit le service correspondant :

```powershell
docker compose up -d postgres
```

## 6. Première préparation de la base de documents

Cette étape est **importante**. Elle prépare le corpus de référence avant le premier test.

Depuis la racine du projet :

```powershell
cd frontend
python manage.py migrate
```

Puis, toujours dans le dossier `frontend` :

```powershell
python manage.py prepare_reference_index
```

### Que fait `prepare_reference_index` ?

Pour chaque document DOCX présent dans le corpus de référence, la commande :

1. extrait le texte ;
2. nettoie et prétraite le texte ;
3. calcule son embedding avec `paraphrase-multilingual-MiniLM-L12-v2` ;
4. enregistre l'embedding dans PostgreSQL ;
5. enregistre également le texte extrait, le texte nettoyé et les métadonnées ;
6. calcule un hash permettant de détecter si le document a changé.

Si un document est déjà indexé et n'a pas changé, son embedding n'est pas recalculé.

### Forcer la réindexation

Si tu veux volontairement recalculer les embeddings de tout le corpus :

```powershell
python manage.py prepare_reference_index --force
```

**Attention :** `--force` est réservé aux besoins de maintenance. Il ne faut pas l'utiliser avant chaque test.

## 7. Première exécution du modèle

Le modèle utilisé est :

`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Lors de sa première utilisation, s'il n'est pas encore présent localement, il est téléchargé depuis Hugging Face.

Il est ensuite sauvegardé localement dans :

`modeles/paraphrase-multilingual-MiniLM-L12-v2/`

Les prochaines exécutions réutilisent cette copie locale.

Un `HF_TOKEN` peut être ajouté dans `frontend/.env` pour authentifier le téléchargement Hugging Face et bénéficier de limites plus élevées.

## 8. Démarrage de l'application

Il faut lancer **deux serveurs**.

### Terminal 1 — API FastAPI

Depuis la racine du projet :

```powershell
uvicorn api.app:app --reload --port 8000
```

### Terminal 2 — application Django

Ouvrir un deuxième terminal, activer l'environnement virtuel puis :

```powershell
cd C:\Users\Leslie\Desktop\Plagiat\detecteur_de_plagiat\frontend
venv\..\venv\Scripts\activate
python manage.py runserver 127.0.0.1:8001
```

Si l'environnement virtuel est déjà actif, utiliser simplement :

```powershell
python manage.py runserver 127.0.0.1:8001
```

## 9. Ordre recommandé pour un nouveau poste

```text
1. Cloner le projet
        ↓
2. Activer le venv
        ↓
3. Installer les dépendances
        ↓
4. Configurer PostgreSQL et frontend/.env
        ↓
5. cd frontend
        ↓
6. python manage.py migrate
        ↓
7. python manage.py prepare_reference_index
        ↓
8. Revenir à la racine
        ↓
9. uvicorn api.app:app --reload --port 8000
        ↓
10. Dans un deuxième terminal : runserver Django
        ↓
11. Effectuer les tests
```

## 10. Ce qui se passe lors des exécutions suivantes

Une fois la préparation terminée, il n'est normalement plus nécessaire de lancer `prepare_reference_index` avant chaque test.

Au démarrage suivant :

- PostgreSQL contient déjà les embeddings des références ;
- le modèle est déjà présent localement ;
- les documents de référence inchangés ne sont pas ré-embeddés ;
- l'index TF-IDF est préparé en mémoire ;
- un upload ne calcule l'embedding que du document candidat ;
- le candidat n'est pas ajouté automatiquement au corpus de référence.

Le temps d'analyse doit donc être nettement inférieur au temps de préparation initiale.

## 11. Calcul du score hybride

Le score final utilise :

```text
30 % × score lexical
+
70 % × score sémantique
=
score hybride
```

### Similarité lexicale

Elle utilise TF-IDF et la similarité cosinus pour mesurer les ressemblances de vocabulaire et de formulations entre le document candidat et les références.

### Similarité sémantique

Elle utilise les embeddings produits par Sentence Transformer puis une similarité cosinus. Elle permet de détecter des formulations différentes mais ayant un sens proche.

### Score hybride

`score_hybride = 0,30 × score_lexical + 0,70 × score_sémantique`

Les seuils actuels sont :

- doublon TDR : `0,70` ;
- plagiat rapport : `0,55`.

Ces seuils devront être calibrés avec des documents annotés avant une utilisation en production.

## 12. Optimisation des temps de calcul

Le système a été conçu pour éviter les opérations coûteuses à chaque upload :

1. le modèle n'est téléchargé qu'une seule fois ;
2. le modèle est réutilisé en mémoire pendant le processus FastAPI ;
3. les embeddings du corpus sont stockés dans PostgreSQL ;
4. les embeddings des références ne sont pas recalculés si le contenu n'a pas changé ;
5. l'index TF-IDF des références est conservé en mémoire ;
6. l'analyse détaillée des passages est limitée aux meilleures références.

## 13. Vérifier l'état du moteur

Une fois FastAPI lancé, consulter :

`http://127.0.0.1:8000/api/health`

Cette route permet notamment de vérifier :

- que l'API fonctionne ;
- que le modèle est disponible ;
- le nombre de références indexées pour les modes `duplicate` et `plagiarism` ;
- les index de références déjà chargés en mémoire.

La documentation interactive de l'API est disponible sur :

`http://127.0.0.1:8000/docs`

## 14. Endpoints principaux

- `POST /api/detect/duplicate` — comparaison avec les TDR de référence ;
- `POST /api/detect/plagiarism` — comparaison avec les rapports de référence ;
- `POST /api/compare` — comparaison directe de deux documents.

## 15. Historique

Les résultats des analyses utilisateurs sont conservés dans l'historique de l'application.

En revanche, un document uploadé n'est **pas automatiquement ajouté** à la base des documents de référence.

## 16. Structure simplifiée

```text
detecteur_de_plagiat/
├── api/
│   └── app.py
├── frontend/
│   ├── manage.py
│   ├── frontend/
│   └── detector/
├── src/
│   ├── extraction/
│   ├── preprocessing/
│   ├── indexing/
│   ├── passages/
│   └── similarity/
├── donnees/
│   ├── TDR/
│   └── Rapport d'etude/
├── modeles/
└── README.md
```

## 17. Branche de développement

Cette évolution est développée **uniquement sur `feature/testsimilarity`**.

`main`, `suite_developpement` et `feature/fusion` ne doivent pas être modifiées dans le cadre de cette évolution.
# FICHE D’IMPLÉMENTATION — DOCSEC

**Détecteur de plagiat et de doublon d’études**  
Dépôt : `ZombreChristian/detecteur_de_plagiat`  
Branche : `main`  
Version API observée : `1.2.0`  
Date : 15 septembre 2026

## 1. Objet de la fiche

Cette fiche décrit l’implémentation réelle du prototype DOCSEC, depuis la constitution du corpus jusqu’à l’intégration du moteur de similarité dans FastAPI et Django. Elle documente les traitements, les fichiers concernés, les données produites, les algorithmes, les règles de décision et le cheminement d’une analyse.

## 2. Finalité fonctionnelle

DOCSEC distingue deux contrôles.

- **Doublon à la programmation :** un TDR est comparé aux TDR existants afin de déterminer s’il constitue un nouveau sujet ou un sujet déjà couvert.
- **Plagiat à la réception :** un rapport d’étude est comparé au corpus de rapports existants. Le contrôle global est complété par une analyse au niveau des passages afin d’identifier les reprises textuelles ou reformulations.

| Contrôle | Document | Corpus | Seuil actuel |
|---|---|---|---|
| Doublon | TDR | `donnees/TDR` | 0,70 |
| Plagiat | Rapport | `donnees/Rapport d'etude` | 0,55 |

## 3. Architecture générale

Le système est organisé en couches :

`DOCX → extraction → nettoyage → TF-IDF + embeddings → scores → score hybride → décision → explication → FastAPI → Django → PostgreSQL`

Principaux composants :

- `src/extraction/extract_docx.py` : extraction DOCX ;
- `src/preprocessing/clean_text.py` : nettoyage ;
- `src/dataset/build_pairs.py` : construction des paires ;
- `src/similarity/tfidf_similarity.py` : baseline lexicale ;
- `src/embeddings/generate_embeddings.py` : embeddings ;
- `src/similarity/semantic_similarity.py` : similarité sémantique et fusion ;
- `src/training/train_model.py` : classifieur ;
- `src/evaluation/evaluate_model.py` : évaluation ;
- `src/prediction/predict.py` : comparaison directe ;
- `src/passages/passage_engine.py` : comparaison par passages ;
- `api/app.py` : moteur FastAPI ;
- `frontend/detector/` : interface, modèles et vues Django.

## 4. Étape 1 — Constitution du corpus

Le corpus actuel contient 120 documents DOCX : 60 TDR et 60 rapports. Les TDR comportent des originaux, des cas proches/doublons et des cas proches mais non doublons. Les rapports comportent des originaux, des cas de réutilisation de contenu, des paraphrases et des cas difficiles de quasi-doublon.

`dataset_metadata.csv` décrit les relations documentaires. Le champ `document_source` sert à relier un document à sa source et à construire des exemples positifs.

## 5. Étape 2 — Extraction DOCX

Fichier : `src/extraction/extract_docx.py`.

Le script utilise `python-docx`. Il parcourt les paragraphes non vides puis les tableaux. Les cellules d’une même ligne sont réunies avec `|`. Chaque document reçoit un identifiant canonique : `TDR-001` à `TDR-060` ou `RAP-001` à `RAP-060`.

La sortie `donnees/documents_extraits.csv` contient notamment `document_id`, `type_document`, `path`, `text`, `nb_caracteres` et `nb_mots`.

Le script vérifie aussi l’unicité des identifiants et le nombre attendu de documents.

## 6. Étape 3 — Nettoyage et normalisation

Fichier : `src/preprocessing/clean_text.py`.

Le nettoyage :

1. convertit les valeurs en chaînes ;
2. applique la normalisation Unicode NFKC ;
3. remplace les espaces insécables ;
4. transforme retours à la ligne et tabulations en espaces ;
5. réduit les espaces multiples ;
6. supprime les espaces en début et fin.

Entrée : `documents_extraits.csv`. Sortie : `documents_nettoyes.csv`. Des statistiques avant/après nettoyage sont conservées.

## 7. Étape 4 — Construction des paires

Fichier : `src/dataset/build_pairs.py`.

Les relations `document_source` valides produisent des paires positives (`label=1`). Les documents sans relation source produisent des paires négatives (`label=0`). Le script privilégie les documents du même secteur ou de la même zone afin de produire des négatifs plus difficiles et plus réalistes.

Sortie : `donnees/pairs.csv`.

## 8. Étape 5 — TF-IDF et similarité cosinus

Fichier : `src/similarity/tfidf_similarity.py`.

Les textes sont transformés en vecteurs TF-IDF avec unigrammes et bigrammes (`ngram_range=(1,2)`) et `sublinear_tf=True`. La proximité est mesurée avec la similarité cosinus.

TF-IDF est efficace lorsque deux documents reprennent une partie importante du même vocabulaire. Sa limite est la paraphrase : deux formulations de même sens peuvent utiliser des mots différents.

Sortie : `resultats/tfidf_predictions.csv`.

## 9. Étape 6 — Embeddings sémantiques

Fichier : `src/embeddings/generate_embeddings.py`.

Le modèle utilisé est `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Chaque document nettoyé est transformé en vecteur dense normalisé. Les identifiants et embeddings sont enregistrés dans `modeles/embeddings.npz`.

Cette représentation permet de mesurer la proximité de sens et améliore la détection de contenus reformulés.

## 10. Étape 7 — Score hybride

Fichier : `src/similarity/semantic_similarity.py`.

Le système calcule le score TF-IDF et le score sémantique, puis les fusionne :

**Score hybride = 0,35 × TF-IDF + 0,65 × similarité sémantique**

La composante sémantique est donc majoritaire, tout en conservant l’information lexicale. La sortie est `resultats/semantic_predictions.csv`.

## 11. Étape 8 — Entraînement du classifieur

Fichier : `src/training/train_model.py`.

Une `LogisticRegression` est entraînée sur `tfidf_score`, `semantic_score` et `hybrid_score`. `class_weight='balanced'` est utilisé. La séparation est effectuée avec `GroupShuffleSplit`, `test_size=0.25` et `random_state=42`, afin de limiter les fuites entre familles de documents.

Le modèle est enregistré dans `modeles/similarity_classifier.joblib`. Un rapport est écrit dans `resultats/metrics/classification_report.txt`.

## 12. Étape 9 — Évaluation

Fichier : `src/evaluation/evaluate_model.py`.

Le modèle est évalué sur un jeu de test séparé. Les métriques calculées sont Accuracy, Precision, Recall, F1-score, ROC-AUC, matrice de confusion et rapport de classification.

La sortie est `resultats/metrics/evaluation_model.txt`.

## 13. Étape 10 — Prédiction directe

Fichier : `src/prediction/predict.py`.

Ce module compare deux DOCX. Il extrait les textes, calcule TF-IDF, la similarité sémantique et le score hybride, charge le classifieur entraîné et produit une probabilité ainsi qu’une décision.

Il s’agit d’un outil de comparaison directe. Il ne faut pas le confondre avec l’endpoint métier `/api/detect/{kind}`, qui applique actuellement directement le seuil au score hybride.

## 14. Étape 11 — Analyse au niveau des passages

Fichier : `src/passages/passage_engine.py`.

Cette couche est principalement utilisée pour le plagiat. Le texte est séparé en paragraphes. Les paragraphes longs sont ensuite découpés en groupes de phrases, avec une limite d’environ 1200 caractères.

Chaque passage candidat est comparé aux passages de la source avec la fusion 35 % TF-IDF / 65 % sémantique. Un passage est signalé lorsque son score atteint 0,50. Le moteur conserve le passage candidat, le passage source, les scores lexical et sémantique et le score global.

La couverture correspond à la proportion de passages du candidat ayant une correspondance retenue.

## 15. Étape 12 — API FastAPI

Fichier : `api/app.py`.

L’API reçoit un DOCX via `UploadFile`, crée un fichier temporaire, extrait son contenu et compare le document au corpus correspondant.

Configuration actuelle :

- `duplicate` → `donnees/TDR` → seuil 0,70 ;
- `plagiarism` → `donnees/Rapport d'etude` → seuil 0,55.

Pour chaque source, TF-IDF et la similarité sémantique sont calculés. Les résultats sont triés par score hybride décroissant. Le meilleur résultat est utilisé pour la décision : `SIMILAIRE` si le score atteint le seuil, sinon `DIFFERENT`.

En mode `plagiarism`, les cinq meilleures sources font ensuite l’objet d’une analyse par passages.

La réponse JSON contient notamment `mode`, `decision`, `threshold`, `best_source`, `tfidf_score`, `semantic_score`, `hybrid_score`, `novelty_score`, `sources` et `matches`.

**Point d’implémentation important :** le classifieur supervisé existe dans le pipeline d’entraînement et dans `predict.py`, mais l’endpoint `/api/detect/{kind}` actuel ne charge pas ce classifieur pour prendre sa décision ; il applique directement les seuils au score hybride.

## 16. Étape 13 — Réception et transfert Django

Fichier : `frontend/detector/views.py`.

`reception()` gère la réception métier du TDR ou du rapport. `analyser()` gère le lancement d’une nouvelle analyse. Les deux utilisent `_get_uploaded_docx()` et `_validate_docx()`.

La fonction `_get_uploaded_docx()` accepte plusieurs noms de champ (`document`, `tdr_document`, `report_document`, `file`) puis recherche dans `request.FILES`. La validation vérifie que le fichier existe et possède l’extension `.docx`.

Le document est envoyé à FastAPI en multipart/form-data.

### Correction importante sur l’upload

Le champ HTML `input type=file` ne doit pas être désactivé avant la soumission. Un champ fichier désactivé n’est pas inclus dans la requête multipart et Django reçoit alors un `request.FILES` vide. La solution appliquée consiste à désactiver uniquement le bouton de soumission et à afficher un indicateur de traitement.

## 17. Étape 14 — PostgreSQL et modèles Django

Le projet utilise PostgreSQL.

Le modèle `Analysis` conserve : utilisateur, nom du document, mode, décision, score hybride, score sémantique, score TF-IDF, nouveauté, résultat JSON complet, statut, message d’erreur, durée et date.

`StudyDocument` représente le registre documentaire structuré : titre, objet, zone géographique, secteur, résultats attendus, entité commanditaire, année, budget, statut, type et chemin du fichier.

`WhitelistedPassage` prépare la gestion de passages administratifs récurrents pouvant être traités comme contenu récurrent.

## 18. Étape 15 — Tableau de bord et historique

`dashboard()` récupère les analyses de l’utilisateur connecté. Il permet une recherche par texte, un filtre par décision et un filtre par mode. La pagination est de 10 analyses par page.

Les statistiques affichent notamment le nombre total d’analyses, les décisions `SIMILAIRE` et `DIFFERENT`, les analyses à examiner et les volumes TDR/rapports.

Chaque document de l’historique est cliquable et ouvre `/analyse/<id>/`.

## 19. Étape 16 — Page de détail

`analysis_detail()` récupère l’analyse demandée et vérifie qu’elle appartient à l’utilisateur connecté.

La page `analysis_detail.html` affiche le nom du document, la date, la décision, le mode, le seuil, les scores TF-IDF/sémantique/hybride, la nouveauté, la durée, la meilleure source et les sources proches. En mode plagiat, les passages signalés sont également affichés.

Une erreur rencontrée sur `/analyse/8/` provenait d’une référence à `source.document`, alors que la réponse FastAPI utilise la clé `source`. Le template a été corrigé pour correspondre à la structure réelle du JSON.

## 20. Étape 17 — Exports

`export_excel()` utilise `openpyxl` et produit un fichier Excel contenant les principales informations de l’analyse et la meilleure source.

`export_pdf()` utilise ReportLab et produit un rapport PDF contenant les informations principales et les sources proches.

## 21. Processus complet — TDR

1. L’utilisateur se connecte.
2. Il sélectionne un TDR DOCX.
3. Django valide le fichier.
4. Django transmet le fichier à FastAPI.
5. FastAPI extrait le texte.
6. Le TDR est comparé aux TDR existants.
7. TF-IDF et Sentence Transformer calculent les similarités.
8. Le score hybride est calculé à 35 % / 65 %.
9. Le meilleur score est comparé au seuil 0,70.
10. `SIMILAIRE` signifie que le TDR est suffisamment proche d’un sujet existant et n’est pas retenu comme nouveau sujet.
11. `DIFFERENT` signifie qu’aucune similarité suffisante n’a été détectée et que le sujet peut poursuivre son processus.
12. Django enregistre le résultat dans PostgreSQL et le rend disponible dans l’historique.

## 22. Processus complet — Rapport

1. Le rapport est soumis après la réalisation de l’étude.
2. Django l’envoie au mode `plagiarism`.
3. FastAPI le compare au corpus des rapports.
4. TF-IDF et la similarité sémantique sont fusionnés.
5. Le seuil global 0,55 est appliqué.
6. Les cinq sources les plus proches sont approfondies.
7. Le moteur par passages recherche les reprises et reformulations à partir de 0,50.
8. Les résultats sont enregistrés dans `Analysis`.
9. Django permet de consulter les sources et les passages dans le détail.

## 23. Chaîne des fichiers

| Étape | Entrée | Sortie |
|---|---|---|
| Extraction | DOCX | `documents_extraits.csv` |
| Nettoyage | `documents_extraits.csv` | `documents_nettoyes.csv` |
| Paires | `dataset_metadata.csv` | `pairs.csv` |
| TF-IDF | documents + paires | `tfidf_predictions.csv` |
| Embeddings | documents nettoyés | `embeddings.npz` |
| Sémantique | documents + paires | `semantic_predictions.csv` |
| Entraînement | prédictions sémantiques | `similarity_classifier.joblib` |
| Évaluation | prédictions + modèle | `evaluation_model.txt` |
| API | DOCX envoyé | JSON d’analyse |
| Django | JSON FastAPI | historique PostgreSQL + interface |

## 24. Technologies utilisées

- Python ;
- python-docx ;
- pandas et NumPy ;
- scikit-learn ;
- Sentence Transformers ;
- FastAPI et Uvicorn ;
- Django ;
- PostgreSQL ;
- openpyxl ;
- ReportLab.

## 25. Sécurité et contrôles applicatifs

- Les vues métier Django utilisent `login_required`.
- Le détail d’une analyse est filtré sur `user=request.user`.
- Les noms de fichiers reçus par FastAPI sont assainis avant création du fichier temporaire.
- `TemporaryDirectory` permet de supprimer les fichiers temporaires après traitement.
- Les paramètres sensibles de Django/PostgreSQL sont prévus via variables d’environnement.

## 26. Vérification et démarrage

1. `git pull origin main`
2. Configurer PostgreSQL et les variables d’environnement.
3. Installer les dépendances Python.
4. Appliquer les migrations Django.
5. Démarrer FastAPI.
6. Démarrer Django.
7. Vérifier `/api/health`.
8. Tester un TDR.
9. Tester un rapport.
10. Tester l’historique, les filtres, la page de détail et les exports.

## 27. État de l’implémentation

La branche `main` contient le pipeline documentaire, le moteur hybride, l’API FastAPI, l’interface Django, PostgreSQL, l’historique, la page de détail, les indicateurs de traitement et les exports. Le classifieur supervisé est disponible comme composant du pipeline expérimental et de comparaison directe, tandis que la décision du endpoint métier actuel repose sur le score hybride et les seuils configurés.

## 28. Principaux fichiers de référence

- `api/app.py`
- `src/extraction/extract_docx.py`
- `src/preprocessing/clean_text.py`
- `src/dataset/build_pairs.py`
- `src/similarity/tfidf_similarity.py`
- `src/embeddings/generate_embeddings.py`
- `src/similarity/semantic_similarity.py`
- `src/training/train_model.py`
- `src/evaluation/evaluate_model.py`
- `src/prediction/predict.py`
- `src/passages/passage_engine.py`
- `frontend/detector/views.py`
- `frontend/detector/models.py`
- `frontend/frontend/settings.py`
- `frontend/templates/dashboard.html`
- `frontend/templates/reception.html`
- `frontend/templates/analyser.html`
- `frontend/templates/analysis_detail.html`

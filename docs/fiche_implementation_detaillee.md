# Fiche d’implémentation détaillée — TDRDOC-SCAN

Ce document complète la fiche Word `Fiche_implementation_TDRDOC-SCAN.docx` avec une description technique approfondie des trois étapes centrales du moteur de similarité.

## 1. Étape 5 — TF-IDF et similarité cosinus

**Fichier :** `src/similarity/tfidf_similarity.py`

### 1.1 Rôle de l’étape

TF-IDF constitue la première mesure quantitative de proximité entre deux documents. Les textes nettoyés provenant de `donnees/documents_nettoyes.csv` sont transformés en vecteurs numériques. Le calcul est ensuite réalisé pour les paires définies dans `donnees/pairs.csv`.

L’objectif est de mesurer la proximité **lexicale** : deux documents qui utilisent une partie importante du même vocabulaire obtiennent généralement une similarité élevée.

### 1.2 Paramétrage réellement utilisé

Le code utilise `TfidfVectorizer` avec les paramètres suivants :

```python
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.98,
    sublinear_tf=True
)
```

- `lowercase=True` convertit le texte en minuscules avant la vectorisation ;
- `ngram_range=(1,2)` utilise les unigrammes et les bigrammes ;
- `min_df=1` conserve les termes apparaissant au moins dans un document ;
- `max_df=0.98` limite l’influence des termes présents dans presque tout le corpus ;
- `sublinear_tf=True` applique une pondération logarithmique à la fréquence des termes.

### 1.3 Unigrammes et bigrammes

Un unigramme est un mot isolé. Un bigramme est une séquence de deux mots. L’utilisation des deux permet de conserver davantage d’information lexicale et une partie de l’ordre local des mots.

Par exemple, un texte contenant « sécurité alimentaire » produit des caractéristiques correspondant notamment à `sécurité`, `alimentaire` et `sécurité alimentaire`.

Cette configuration est utile pour les documents administratifs, car certaines expressions composées portent davantage d’information que les mots pris séparément.

### 1.4 Calcul TF

La fréquence TF mesure la présence d’un terme dans un document. Avec `sublinear_tf=True`, la fréquence brute est compressée logarithmiquement. Pour une fréquence positive, la pondération utilisée est de la forme :

```text
TF = 1 + log(tf)
```

Cette transformation évite qu’un mot répété un très grand nombre de fois domine excessivement le vecteur.

### 1.5 Calcul IDF

La composante IDF réduit l’importance des mots présents dans de nombreux documents et augmente relativement celle des termes plus discriminants.

Le principe est :

```text
TF-IDF = TF × IDF
```

Ainsi, un terme très fréquent dans un document mais également présent dans presque tout le corpus apporte moins d’information qu’un terme plus spécifique.

### 1.6 Construction de la matrice

Après vectorisation, chaque document devient un vecteur dans un espace de caractéristiques. La matrice est construite par :

```python
X = vectorizer.fit_transform(docs["text"].fillna(""))
```

Le script conserve une correspondance entre `document_id` et position dans cette matrice afin de retrouver rapidement le vecteur associé à `doc_a` ou `doc_b`.

### 1.7 Similarité cosinus

Les vecteurs TF-IDF de deux documents sont comparés avec la similarité cosinus :

```text
cos(A,B) = (A · B) / (||A|| × ||B||)
```

Dans le code :

```python
score = float(
    cosine_similarity(
        X[positions[a]],
        X[positions[b]]
    )[0, 0]
)
```

Le résultat représente la proximité de direction entre les deux vecteurs. Pour cette application, plus le score est élevé, plus le contenu lexical est proche.

### 1.8 Production du résultat

Le score est ajouté à la colonne `tfidf_score` de `pairs.csv`, puis le résultat complet est enregistré dans :

```text
resultats/tfidf_predictions.csv
```

Le fichier contient notamment `doc_a`, `doc_b`, `label` et `tfidf_score`.

### 1.9 Limite de TF-IDF

TF-IDF ne comprend pas réellement le sens. Une phrase peut être reformulée avec des mots différents tout en conservant la même idée. Dans ce cas, la similarité lexicale peut rester modérée. C’est précisément pour cette raison que TDRDOC-SCAN ajoute les embeddings sémantiques.

---

## 2. Étape 6 — Embeddings sémantiques

**Fichier :** `src/embeddings/generate_embeddings.py`

### 2.1 Objectif

L’objectif est de compléter la comparaison lexicale par une représentation capable de rapprocher des formulations différentes mais de sens voisin.

Le modèle réellement configuré est :

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### 2.2 Chaîne de transformation

Le traitement peut être résumé ainsi :

```text
Texte nettoyé
    ↓
Tokenisation
    ↓
Encodeur Transformer
    ↓
Représentation contextualisée
    ↓
Embedding dense
    ↓
Normalisation
```

Le code réalise :

```python
embeddings = model.encode(
    docs["text"].fillna("").tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)
```

### 2.3 Pourquoi la normalisation ?

`normalize_embeddings=True` normalise les vecteurs. Cette propriété facilite leur comparaison par produit scalaire. Lorsque les vecteurs sont normalisés, leur produit scalaire correspond à leur similarité cosinus.

Dans `semantic_similarity.py`, le calcul est réalisé ainsi :

```python
semantic_score = np.dot(
    emb[epos[r.doc_a]],
    emb[epos[r.doc_b]]
)
```

### 2.4 Intérêt par rapport à TF-IDF

TF-IDF répond principalement à la question : « les deux documents utilisent-ils les mêmes termes ? »

L’embedding cherche plutôt à représenter : « les deux documents parlent-ils de choses proches ? »

Cette différence est importante pour les reformulations. Par exemple, les formulations « renforcer le suivi des infrastructures hydrauliques » et « améliorer la surveillance des ouvrages d’eau » ne reposent pas exactement sur les mêmes mots, mais leur contenu peut être rapproché sémantiquement.

### 2.5 Stockage

`generate_embeddings.py` enregistre les identifiants et les vecteurs dans :

```text
modeles/embeddings.npz
```

Le fichier permet de conserver la représentation vectorielle produite pour le corpus.

### 2.6 Limites

Une forte proximité sémantique n’est pas automatiquement une preuve de plagiat. Deux études légitimes portant sur le même domaine peuvent naturellement utiliser des concepts et des formulations proches. L’embedding est donc un signal de similarité, pas une conclusion juridique ou scientifique.

---

## 3. Étape 7 — Score hybride

**Fichier :** `src/similarity/semantic_similarity.py` et logique équivalente dans `api/app.py`

### 3.1 Principe

Le score hybride combine les deux mesures précédentes :

1. le score lexical TF-IDF ;
2. le score sémantique des embeddings.

L’objectif est de conserver la capacité de détecter les reprises lexicales tout en améliorant la détection des reformulations.

### 3.2 Formule réellement implémentée

Le projet utilise actuellement :

```text
Score hybride = 0,35 × score TF-IDF
              + 0,65 × score sémantique
```

La composante sémantique représente donc 65 % du score et la composante lexicale 35 %.

Dans `semantic_similarity.py` :

```python
pairs["hybrid_score"] = (
    0.35 * pairs.tfidf_score
    + 0.65 * pairs.semantic_score
)
```

Dans `api/app.py`, la même fusion est appliquée à chaque document source :

```python
hybrid = float(
    0.35 * lexical[i]
    + 0.65 * semantic[i]
)
```

### 3.3 Exemple numérique réel

Pour une comparaison où :

```text
TF-IDF = 0,6616
Sémantique = 0,8657
```

le score est :

```text
0,35 × 0,6616 + 0,65 × 0,8657
= 0,23156 + 0,562705
= 0,794265
≈ 0,7943
```

Le score hybride obtenu est donc **0,7943**.

### 3.4 Passage du score à la décision

Le score hybride est d’abord calculé pour toutes les sources du corpus. Les sources sont ensuite triées par ordre décroissant. Le meilleur résultat devient la source la plus proche.

La décision est ensuite appliquée séparément :

```text
duplicate / TDR : SIMILAIRE si score ≥ 0,70
plagiarism / rapport : SIMILAIRE si score ≥ 0,55
sinon : DIFFERENT
```

Le seuil n’est donc pas utilisé pour calculer le score ; il est utilisé après le calcul pour transformer une mesure continue en décision métier binaire.

### 3.5 Pourquoi la décision utilise le meilleur document ?

Dans le parcours métier, le candidat est comparé à l’ensemble du corpus. Il serait insuffisant de comparer le candidat à un seul document arbitraire. Le système calcule donc un score pour chaque source disponible, classe les résultats et retient la source présentant le score hybride maximal.

Ainsi, pour un candidat donné :

```text
Candidat
  ↓
Comparaison avec TDR-001
Comparaison avec TDR-002
Comparaison avec TDR-003
...
Comparaison avec TDR-060
  ↓
Classement des scores hybrides
  ↓
Meilleure source
  ↓
Application du seuil
  ↓
SIMILAIRE / DIFFERENT
```

### 3.6 Pourquoi 35 % / 65 % ?

La pondération actuelle donne davantage d’importance à la similarité sémantique afin de mieux prendre en compte les reformulations. Le TF-IDF reste toutefois présent pour conserver un signal directement lié au vocabulaire effectivement partagé.

Cette pondération est un **paramètre du prototype**. Elle ne doit pas être présentée comme une valeur universelle. Une validation expérimentale plus large pourrait comparer plusieurs combinaisons, par exemple différentes pondérations lexicales/sémantiques, puis retenir celle qui offre le meilleur compromis sur un jeu de validation représentatif.

### 3.7 Distinction entre score et décision

Il est essentiel de distinguer les trois niveaux :

| Niveau | Signification |
|---|---|
| TF-IDF | Proximité lexicale entre deux textes |
| Sémantique | Proximité de sens représentée par les embeddings |
| Hybride | Combinaison pondérée des deux signaux |
| Décision | Application d’un seuil au meilleur score hybride |

Le score hybride n’est donc pas lui-même une décision. Il devient une décision uniquement après comparaison avec le seuil correspondant au mode.

### 3.8 Relation avec le classifieur supervisé

Le projet contient également une `LogisticRegression` entraînée sur `tfidf_score`, `semantic_score` et `hybrid_score`. Cependant, l’endpoint métier actuel `/api/detect/{kind}` ne charge pas ce classifieur pour décider. Il applique directement le seuil au score hybride.

Cette distinction doit être conservée dans la documentation scientifique :

```text
Pipeline expérimental : scores → LogisticRegression → prédiction

Parcours métier actuel : scores → score hybride → seuil → décision
```

### 3.9 Sorties

Le pipeline expérimental écrit notamment :

```text
resultats/tfidf_predictions.csv
modeles/embeddings.npz
resultats/semantic_predictions.csv
```

L’API métier retourne ensuite dans son JSON :

```text
mode
 decision
 threshold
 best_source
 tfidf_score
 semantic_score
 hybrid_score
 novelty_score
 sources
 matches
```

---

## 4. Interprétation globale des trois étapes

Les trois étapes forment une chaîne cohérente :

```text
                 ┌─────────────────┐
Texte nettoyé →  │      TF-IDF     │ → score lexical
                 └─────────────────┘
                          │
                          ├──────────────┐
                          │              │
                 ┌─────────────────┐    │
Texte nettoyé →  │    Embedding    │ → score sémantique
                 └─────────────────┘    │
                          │              │
                          └──────┬───────┘
                                 ↓
                         Score hybride
                         35 % / 65 %
                                 ↓
                         Classement des
                             sources
                                 ↓
                         Application du
                             seuil
                                 ↓
                       SIMILAIRE / DIFFERENT
```

Pour le TDR, le meilleur score hybride sert à déterminer si le sujet est suffisamment proche d’un sujet existant. Pour le rapport, le score global est complété par l’analyse par passages afin de rendre les correspondances plus explicables.

## 5. Limites à conserver dans la documentation

- Le corpus actuellement utilisé est interne au projet ; une absence de correspondance ne prouve pas l’originalité sur Internet.
- Une proximité sémantique peut être légitime lorsque deux documents traitent du même domaine.
- Une paraphrase très éloignée peut encore échapper à la détection.
- Les seuils `0,70` et `0,55` sont des paramètres actuels du prototype et doivent être validés sur des données représentatives.
- La pondération `35 % / 65 %` est également un paramètre expérimental.
- Le classifieur supervisé existe, mais la décision de l’endpoint métier actuel repose directement sur le score hybride et le seuil.
- Le système fournit un indicateur automatisé de proximité ; il ne constitue pas à lui seul une preuve juridique de plagiat.

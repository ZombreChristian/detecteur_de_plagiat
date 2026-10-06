import re
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.similarity.model_manager import get_model


FRENCH_STOPWORDS = {
    "alors", "au", "aucun", "aussi", "autre", "avec", "avoir", "avant",
    "aux", "car", "ce", "ceci", "cela", "ces", "cet", "cette", "comme",
    "dans", "de", "des", "du", "elle", "elles", "en", "entre", "est",
    "et", "eux", "il", "ils", "je", "la", "le", "les", "leur", "leurs",
    "lui", "mais", "me", "mes", "meme", "mêmes", "mon", "ne", "nos",
    "notre", "nous", "on", "ont", "ou", "par", "pas", "pour", "que",
    "quel", "quelle", "quelles", "quels", "qui", "sa", "sans", "se",
    "sera", "seront", "ses", "soi", "soit", "sont", "sur", "ta", "te",
    "tes", "toi", "ton", "tous", "tout", "toute", "toutes", "un", "une",
    "vos", "votre", "vous", "y", "afin", "ainsi", "apres", "après",
    "chez", "dont", "peut", "peuvent", "plus", "moins", "tres", "très",
    "doit", "doivent", "sera", "serait", "etre", "être",
}


def normalize_passage(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _fold_text(text):
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _topic_tokens(text):
    folded = _fold_text(normalize_passage(text))
    tokens = re.findall(r"[a-zà-ÿ]{4,}", folded)
    return {token for token in tokens if token not in FRENCH_STOPWORDS}


def split_into_passages(text, max_chars=1200):
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n+", text or "")
        if p.strip()
    ]

    passages = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            passages.append(paragraph)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in sentences:
            if current and len(current) + len(sentence) + 1 > max_chars:
                passages.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            passages.append(current)

    return passages


def adjusted_hybrid_score(
    candidate,
    source,
    lexical,
    semantic,
    lexical_weight=0.30,
    semantic_weight=0.70,
):
    """Score hybride simple : 30 % TF-IDF + 70 % sémantique."""
    return float(
        np.clip(
            lexical_weight * float(lexical)
            + semantic_weight * float(semantic),
            0.0,
            1.0,
        )
    )


PASSAGE_MATCH_MIN_SEMANTIC = 0.58


def compare_passages(
    candidate_passages,
    source_passages,
    top_k=10,
    lexical_weight=0.30,
    semantic_weight=0.70,
):
    """
    La similarité sémantique d'un passage est uniquement la cosine similarity
    entre les embeddings du passage candidat et du passage source.

    Le TF-IDF est calculé séparément et ne modifie pas le score sémantique.
    """
    if not candidate_passages or not source_passages:
        return []

    texts = candidate_passages + source_passages

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
    ).fit_transform(texts)

    lexical = cosine_similarity(
        tfidf[:len(candidate_passages)],
        tfidf[len(candidate_passages):],
    )

    embeddings = get_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    semantic = np.matmul(
        embeddings[:len(candidate_passages)],
        embeddings[len(candidate_passages):].T,
    )
    semantic = np.clip(semantic, 0.0, 1.0)

    candidate_edges = []

    for i, candidate in enumerate(candidate_passages):
        candidate_words = normalize_passage(candidate).split()
        semantic_order = np.argsort(semantic[i])[::-1]

        for j in semantic_order[:5]:
            semantic_score = float(semantic[i, j])
            lexical_score = float(lexical[i, j])

            # La sémantique décide de la correspondance du passage.
            # Les passages très courts exigent une confiance plus forte.
            if len(candidate_words) < 7 and semantic_score < 0.85:
                continue

            if semantic_score < PASSAGE_MATCH_MIN_SEMANTIC:
                continue

            candidate_edges.append({
                "candidate_index": int(i),
                "source_index": int(j),
                "tfidf_score": lexical_score,
                "semantic_score": semantic_score,
            })

    # Une correspondance ne peut utiliser deux fois le même passage.
    candidate_edges.sort(
        key=lambda item: (
            item["semantic_score"],
            item["tfidf_score"],
        ),
        reverse=True,
    )

    used_candidates = set()
    used_sources = set()
    matches = []

    for edge in candidate_edges:
        i = edge["candidate_index"]
        j = edge["source_index"]

        if i in used_candidates or j in used_sources:
            continue

        matches.append({
            "candidate_passage": candidate_passages[i],
            "source_passage": source_passages[j],
            "tfidf_score": round(edge["tfidf_score"], 4),
            "semantic_score": round(edge["semantic_score"], 4),
            "score": round(edge["semantic_score"], 4),
        })

        used_candidates.add(i)
        used_sources.add(j)

        if len(matches) >= top_k:
            break

    matches.sort(key=lambda item: item["semantic_score"], reverse=True)
    return matches


def aggregate_passage_scores(
    matches,
    candidate_passages,
    document_lexical_score=None,
    lexical_weight=0.30,
    semantic_weight=0.70,
):
    """Calcule séparément la sémantique, le TF-IDF puis le score hybride."""
    coverage = calculate_coverage(candidate_passages, matches)

    if not matches:
        lexical_score = float(
            document_lexical_score if document_lexical_score is not None else 0.0
        )
        return {
            "tfidf_score": float(np.clip(lexical_score, 0.0, 1.0)),
            "semantic_score": 0.0,
            "hybrid_score": 0.0,
            "coverage": coverage,
        }

    semantic_mean = float(np.mean([
        float(match["semantic_score"])
        for match in matches
    ]))

    if document_lexical_score is None:
        lexical_score = float(np.mean([
            float(match["tfidf_score"])
            for match in matches
        ]))
    else:
        lexical_score = float(document_lexical_score)

    # La couverture intervient uniquement dans le score sémantique du document.
    coverage_factor = float(np.sqrt(max(0.0, min(1.0, coverage))))
    semantic_score = semantic_mean * coverage_factor

    hybrid_score = (
        lexical_weight * lexical_score
        + semantic_weight * semantic_score
    )

    return {
        "tfidf_score": round(float(np.clip(lexical_score, 0.0, 1.0)), 4),
        "semantic_score": round(float(np.clip(semantic_score, 0.0, 1.0)), 4),
        "hybrid_score": round(float(np.clip(hybrid_score, 0.0, 1.0)), 4),
        "coverage": coverage,
    }


def calculate_coverage(candidate_passages, matches):
    if not candidate_passages:
        return 0.0

    matched = {
        normalize_passage(match["candidate_passage"])
        for match in matches
    }

    return round(len(matched) / len(candidate_passages), 4)


def analyze_document_pair(
    candidate_text,
    source_text,
    top_k=10,
    lexical_weight=0.30,
    semantic_weight=0.70,
    document_lexical_score=None,
):
    candidate_passages = split_into_passages(candidate_text)
    source_passages = split_into_passages(source_text)

    matches = compare_passages(
        candidate_passages,
        source_passages,
        top_k=top_k,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
    )

    scores = aggregate_passage_scores(
        matches,
        candidate_passages,
        document_lexical_score=document_lexical_score,
        lexical_weight=lexical_weight,
        semantic_weight=semantic_weight,
    )

    return {
        "matches": matches,
        "coverage": scores["coverage"],
        "tfidf_score": scores["tfidf_score"],
        "semantic_score": scores["semantic_score"],
        "hybrid_score": scores["hybrid_score"],
        "passages_candidate": len(candidate_passages),
        "passages_source": len(source_passages),
    }

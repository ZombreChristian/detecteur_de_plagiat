from functools import lru_cache
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(MODEL_NAME)


def normalize_passage(text):
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    return text


def split_into_passages(text, max_chars=1200):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text or "") if p.strip()]
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


def compare_passages(candidate_passages, source_passages, threshold=0.50, top_k=10):
    if not candidate_passages or not source_passages:
        return []

    texts = candidate_passages + source_passages
    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    lexical = cosine_similarity(
        tfidf[:len(candidate_passages)],
        tfidf[len(candidate_passages):],
    )
    embeddings = get_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    semantic = np.matmul(
        embeddings[:len(candidate_passages)],
        embeddings[len(candidate_passages):].T,
    )

    matches = []
    for i, candidate in enumerate(candidate_passages):
        combined = 0.35 * lexical[i] + 0.65 * semantic[i]
        # Ignore very short generic headings unless lexical overlap is also strong.
        candidate_words = normalize_passage(candidate).split()
        for j in np.argsort(combined)[::-1][:3]:
            score = float(combined[j])
            lexical_score = float(lexical[i, j])
            semantic_score = float(semantic[i, j])
            source = source_passages[j]
            source_words = normalize_passage(source).split()
            if len(candidate_words) < 7 and score < 0.85:
                continue
            if score >= threshold:
                matches.append({
                    "candidate_passage": candidate,
                    "source_passage": source,
                    "tfidf_score": round(lexical_score, 4),
                    "semantic_score": round(semantic_score, 4),
                    "score": round(score, 4),
                })
                break

    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[:top_k]


def calculate_coverage(candidate_passages, matches):
    if not candidate_passages:
        return 0.0
    matched = {normalize_passage(m["candidate_passage"]) for m in matches}
    return round(len(matched) / len(candidate_passages), 4)


def analyze_document_pair(candidate_text, source_text, threshold=0.50, top_k=10):
    candidate_passages = split_into_passages(candidate_text)
    source_passages = split_into_passages(source_text)
    matches = compare_passages(candidate_passages, source_passages, threshold, top_k)
    coverage = calculate_coverage(candidate_passages, matches)
    return {
        "matches": matches,
        "coverage": coverage,
        "passages_candidate": len(candidate_passages),
        "passages_source": len(source_passages),
    }

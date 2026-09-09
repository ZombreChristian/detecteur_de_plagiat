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

def split_into_passages(text, max_chars=1200):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    passages = []
    for p in paragraphs:
        if len(p) <= max_chars:
            passages.append(p)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            current = ""
            for s in sentences:
                if current and len(current) + len(s) + 1 > max_chars:
                    passages.append(current.strip()); current = s
                else:
                    current = f"{current} {s}".strip()
            if current: passages.append(current)
    return passages

def compare_passages(candidate_passages, source_passages, threshold=0.50, top_k=10):
    if not candidate_passages or not source_passages:
        return []
    texts = candidate_passages + source_passages
    tfidf = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True).fit_transform(texts)
    lexical = cosine_similarity(tfidf[:len(candidate_passages)], tfidf[len(candidate_passages):])
    emb = get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    semantic = np.matmul(emb[:len(candidate_passages)], emb[len(candidate_passages):].T)
    matches = []
    for i, cp in enumerate(candidate_passages):
        j = int(np.argmax(0.35 * lexical[i] + 0.65 * semantic[i]))
        score = float(0.35 * lexical[i, j] + 0.65 * semantic[i, j])
        if score >= threshold:
            matches.append({"candidate_passage": cp, "source_passage": source_passages[j], "tfidf_score": round(float(lexical[i,j]),4), "semantic_score": round(float(semantic[i,j]),4), "score": round(score,4)})
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:top_k]

def calculate_coverage(candidate_passages, matches):
    if not candidate_passages: return 0.0
    matched = {m["candidate_passage"] for m in matches}
    return round(len(matched) / len(candidate_passages), 4)

def analyze_document_pair(candidate_text, source_text, threshold=0.50, top_k=10):
    cp = split_into_passages(candidate_text)
    sp = split_into_passages(source_text)
    matches = compare_passages(cp, sp, threshold, top_k)
    coverage = calculate_coverage(cp, matches)
    return {"matches": matches, "coverage": coverage, "passages_candidate": len(cp), "passages_source": len(sp)}

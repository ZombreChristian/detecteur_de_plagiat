import re
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.similarity.model_manager import get_model


# Formulations très fréquentes dans les TDR. Elles décrivent la façon de conduire
# une étude mais ne suffisent pas, à elles seules, à établir que deux études
# portent sur le même sujet.
GENERIC_PREFIXES = (
    "collect",
    "analys",
    "trait",
    "restit",
    "valid",
    "methodolog",
    "document",
    "recommand",
    "rapport",
    "livr",
    "suiv",
    "comit",
    "reun",
    "entreti",
    "enquet",
    "questionnair",
    "donne",
    "resultat",
    "conclusion",
    "bibliograph",
    "revue",
    "echantillon",
    "echantillonn",
    "indicateur",
    "tableau",
    "graphique",
    "calendrier",
    "chronogramm",
    "mission",
    "prestataire",
    "consultant",
    "restitution",
)

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
    "chez", "dont", "leurs", "peut", "peuvent", "plus", "moins", "tres",
    "très", "doit", "doivent", "sera", "serait", "etre", "être",
}


def normalize_passage(text):
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    return text


def _fold_text(text):
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _topic_tokens(text):
    folded = _fold_text(normalize_passage(text))
    tokens = re.findall(r"[a-zà-ÿ]{4,}", folded)
    result = set()
    for token in tokens:
        if token in FRENCH_STOPWORDS:
            continue
        if any(token.startswith(prefix) for prefix in GENERIC_PREFIXES):
            continue
        result.add(token)
    return result


def _generic_ratio(text):
    folded = _fold_text(normalize_passage(text))
    tokens = re.findall(r"[a-zà-ÿ]{4,}", folded)
    if not tokens:
        return 0.0
    generic = sum(
        1
        for token in tokens
        if token in FRENCH_STOPWORDS
        or any(token.startswith(prefix) for prefix in GENERIC_PREFIXES)
    )
    return generic / len(tokens)


def _topic_overlap(candidate, source):
    candidate_topics = _topic_tokens(candidate)
    source_topics = _topic_tokens(source)
    if not candidate_topics or not source_topics:
        return 0.0

    intersection = len(candidate_topics & source_topics)
    # F1 entre les deux ensembles de termes spécifiques : il évite qu'un
    # document très long soit avantagé simplement parce qu'il contient plus de mots.
    precision = intersection / len(candidate_topics)
    recall = intersection / len(source_topics)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _passage_relevance(candidate, source, semantic_score):
    """Évalue la pertinence thématique d'un match sémantique."""
    topic = _topic_overlap(candidate, source)
    generic = (_generic_ratio(candidate) + _generic_ratio(source)) / 2

    # La proximité thématique conserve l'essentiel du score. Les formulations
    # très génériques sont progressivement pénalisées.
    topic_factor = 0.62 + 0.38 * topic
    generic_excess = max(0.0, generic - 0.30)
    generic_factor = 1.0 - min(0.32, generic_excess * 0.90)

    relevance = float(semantic_score) * topic_factor * generic_factor
    return float(np.clip(relevance, 0.0, 1.0)), topic, generic


def adjusted_hybrid_score(candidate, source, lexical, semantic, lexical_weight=0.30, semantic_weight=0.70):
    """Calcule le score hybride utilisé pour le classement et la décision.

    Les scores lexical et sémantique restent les mesures brutes affichées.
    Le score hybride est corrigé uniquement pour éviter que des formulations
    méthodologiques génériques produisent artificiellement une forte proximité.
    """
    base = lexical_weight * float(lexical) + semantic_weight * float(semantic)
    topic = _topic_overlap(candidate, source)
    generic = (_generic_ratio(candidate) + _generic_ratio(source)) / 2

    # Les documents partageant un vocabulaire thématique spécifique conservent
    # presque tout leur score. À l'inverse, une forte proximité essentiellement
    # générique est progressivement réduite.
    specificity = 0.60 + 0.40 * topic
    generic_penalty = 1.0 - min(0.20, max(0.0, generic - 0.35) * 0.55)

    adjusted = base * specificity * generic_penalty

    # Une très forte similarité lexicale reste un signal solide : cette règle
    # évite de dégrader les vrais doublons quasi identiques.
    if lexical >= 0.88 and semantic >= 0.88:
        adjusted = max(adjusted, base * 0.95)

    return float(np.clip(adjusted, 0.0, 1.0))


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


def compare_passages(
    candidate_passages,
    source_passages,
    semantic_threshold=0.58,
    top_k=10,
    lexical_weight=0.30,
    semantic_weight=0.70,
):
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
        convert_to_numpy=True,
    )
    semantic = np.matmul(
        embeddings[:len(candidate_passages)],
        embeddings[len(candidate_passages):].T,
    )
    semantic = np.clip(semantic, 0.0, 1.0)

    matches = []
    for i, candidate in enumerate(candidate_passages):
        # La détection d'une correspondance sémantique est indépendante du
        # TF-IDF : une paraphrase peut donc être reconnue même avec peu de mots
        # en commun.
        candidate_words = normalize_passage(candidate).split()
        semantic_order = np.argsort(semantic[i])[::-1]

        # On examine plusieurs voisins sémantiques et on retient celui qui
        # combine le mieux proximité sémantique et pertinence thématique.
        best_match = None
        for j in semantic_order[:5]:
            semantic_score = float(semantic[i, j])
            lexical_score = float(lexical[i, j])
            source = source_passages[j]

            if len(candidate_words) < 7 and semantic_score < 0.85:
                continue
            if semantic_score < semantic_threshold:
                continue

            relevance, topic_overlap, generic_ratio = _passage_relevance(
                candidate, source, semantic_score
            )

            # On ne laisse pas un passage très générique devenir une preuve
            # simplement parce que sa similarité sémantique brute est élevée.
            if relevance < 0.50:
                continue

            current = (relevance, semantic_score)
            if best_match is None or current > best_match[0]:
                best_match = (
                    current,
                    j,
                    lexical_score,
                    semantic_score,
                    topic_overlap,
                    generic_ratio,
                )

        if best_match is not None:
            _, j, lexical_score, semantic_score, topic_overlap, generic_ratio = best_match
            matches.append({
                "candidate_passage": candidate,
                "source_passage": source_passages[j],
                "tfidf_score": round(lexical_score, 4),
                "semantic_score": round(semantic_score, 4),
                "topic_overlap": round(topic_overlap, 4),
                "generic_ratio": round(generic_ratio, 4),
                "relevance_score": round(best_match[0][0], 4),
                # Le score affiché reste la proximité sémantique brute.
                "score": round(semantic_score, 4),
            })

    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[:top_k]


def aggregate_passage_scores(
    matches,
    candidate_passages,
    document_lexical_score=None,
    lexical_weight=0.30,
    semantic_weight=0.70,
):
    """Combine un TF-IDF documentaire indépendant avec la sémantique des passages.

    Le TF-IDF ne sert pas de filtre pour détecter une correspondance sémantique.
    """
    if not matches:
        return {
            "tfidf_score": 0.0,
            "semantic_score": 0.0,
            "hybrid_score": 0.0,
            "coverage": 0.0,
        }

    semantic_scores = [float(match["semantic_score"]) for match in matches]
    semantic_score = float(np.mean(semantic_scores))

    if document_lexical_score is None:
        lexical_scores = [float(match["tfidf_score"]) for match in matches]
        lexical_score = float(np.mean(lexical_scores))
    else:
        lexical_score = float(document_lexical_score)

    hybrid_score = float(
        lexical_weight * lexical_score + semantic_weight * semantic_score
    )
    coverage = calculate_coverage(candidate_passages, matches)

    return {
        "tfidf_score": round(float(np.clip(lexical_score, 0.0, 1.0)), 4),
        "semantic_score": round(float(np.clip(semantic_score, 0.0, 1.0)), 4),
        "hybrid_score": round(float(np.clip(hybrid_score, 0.0, 1.0)), 4),
        "coverage": coverage,
    }


def calculate_coverage(candidate_passages, matches):
    if not candidate_passages:
        return 0.0
    matched = {normalize_passage(m["candidate_passage"]) for m in matches}
    return round(len(matched) / len(candidate_passages), 4)


def analyze_document_pair(
    candidate_text,
    source_text,
    threshold=0.58,
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
        semantic_threshold=threshold,
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

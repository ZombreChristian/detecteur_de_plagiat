"""Arbitrage RAG des TDR par Qwen3 via Ollama local.

La recherche des références est effectuée par le moteur existant. Ce module
transmet au modèle le TDR à analyser et les références récupérées; il ne
remplace ni l'index PostgreSQL ni le moteur de récupération.
"""
import json
import os
import re

import requests


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TIMEOUT = max(300.0, float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300")))
# Qwen3 tourne localement avec un contexte souvent limite a 4096 tokens.
# Borner le prompt evite d'envoyer plusieurs dizaines de milliers de caracteres.
RAG_REFERENCE_TOP_K = min(5, max(1, int(os.getenv("TDR_RAG_REFERENCE_TOP_K", "4"))))
MAX_CANDIDATE_CHARS = min(9000, max(3000, int(os.getenv("TDR_RAG_CANDIDATE_MAX_CHARS", "8000"))))
MAX_REFERENCE_CHARS = min(3000, max(1200, int(os.getenv("TDR_RAG_REFERENCE_MAX_CHARS", "2400"))))


class OllamaRAGError(RuntimeError):
    """Le service Ollama ou sa réponse n'est pas exploitable."""


_SYSTEM_PROMPT = """Tu es un expert chargé d'évaluer si un nouveau terme de référence (TDR)
correspond à une étude déjà réalisée parmi les références fournies.
Tu fais une analyse RAG : tu t'appuies uniquement sur le TDR candidat et sur les
extraits/références récupérés. Les scores de recherche servent seulement à retrouver
des références, jamais à décider du résultat.

Compare le sens, même si les formulations diffèrent, et distingue notamment :
- objet réel de l'étude et problématique;
- secteur et activités;
- zone géographique (une zone différente est une différence importante);
- objectifs et résultats attendus;
- population cible, période et méthodologie, comme critères complémentaires.
Ne confonds pas la structure administrative générique des TDR avec leur identité.
N'invente jamais d'information absente. Si les preuves sont insuffisantes, indique-le.

Choisis exactement un cas :
CAS_1_CORRESPONDANCE_FORTE : une référence couvre essentiellement la même étude
(même objet et convergence substantielle des dimensions clés).
CAS_2_CORRESPONDANCE_PARTIELLE : plusieurs dimensions importantes convergent, mais
des différences ou incertitudes empêchent de conclure à un doublon fort.
CAS_3_MEME_OBJECTIF_ZONE_DIFFERENTE : objectifs/objet comparables, mais la zone
géographique étudiée est explicitement différente.
CAS_4_AUCUNE_ETUDE_PERTINENTE : aucune référence récupérée n'est suffisamment
pertinente au regard de l'objet et des objectifs du TDR.

Réponds exclusivement avec un objet JSON valide, sans Markdown, ayant ces clés :
case_id (une des quatre valeurs exactes ci-dessus),
decision (une des valeurs exactes SIMILAIRE, SIMILARITÉ PARTIELLE, DIFFERENT, À EXAMINER),
confidence (FORTE, MOYENNE ou FAIBLE),
case_title (titre court en français),
matched_reference (nom exact d'une référence pertinente parmi celles fournies, ou chaîne vide si aucune ne correspond),
explanation (explication claire en français, 2 à 5 phrases),
common_points (tableau de chaînes),
differences (tableau de chaînes),
evidence (tableau d'objets avec source, point, justification),
missing_information (tableau de chaînes),
recommendation (une recommandation concrète en français).
Le cas 1 correspond à decision SIMILAIRE; le cas 2 à SIMILARITÉ PARTIELLE;
le cas 3 et le cas 4 correspondent à DIFFERENT. Si l'information manque réellement,
choisis À EXAMINER plutôt que d'inventer une conclusion.
Ne cite que les références incluses dans le contexte et donne leurs noms exacts.
Les notes de scores éventuelles sont des indices de récupération, pas une preuve
suffisante de l'identité des études."""


def _response_json(content):
    text = (content or "").strip()
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.S)
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError as exc:
                raise OllamaRAGError("Qwen3 a renvoyé un JSON invalide.") from exc
        raise OllamaRAGError("Qwen3 n'a pas renvoyé de résultat JSON exploitable.")


def _reference_excerpt(candidate_text, source_text, limit):
    """Sélectionne des paragraphes utiles au TDR sans tronquer uniquement le début."""
    source_text = (source_text or "").strip()
    if len(source_text) <= limit:
        return source_text

    candidate_terms = {
        term for term in re.findall(r"[\wÀ-ÿ]{4,}", (candidate_text or "").casefold())
        if term not in {
            "dans", "pour", "avec", "cette", "étude", "etude", "objectif",
            "objectifs", "résultats", "resultats", "termes", "référence",
            "reference", "projet", "ainsi", "entre", "leur", "leurs", "être",
            "etre", "plus", "tout", "tous", "comme", "afin", "selon",
        }
    }
    blocks = [block.strip() for block in re.split(r"\n{2,}|(?<=[.!?])\s+(?=[A-ZÀ-Ÿ])", source_text) if block.strip()]
    if not blocks:
        return source_text[:limit]

    ranked_blocks = []
    for index, block in enumerate(blocks):
        terms = set(re.findall(r"[\wÀ-ÿ]{4,}", block.casefold()))
        overlap = len(candidate_terms & terms)
        # Favoriser les paragraphes de taille raisonnable et les premières sections,
        # sans supposer que le début du document contient toutes les informations utiles.
        score = overlap / max(1, len(terms) ** 0.5) + (0.15 if index < 3 else 0)
        ranked_blocks.append((score, index, block))

    selected = []
    used = 0
    for _, index, block in sorted(ranked_blocks, key=lambda item: (item[0], -item[1]), reverse=True):
        remaining = limit - used
        if remaining <= 0:
            break
        snippet = block if len(block) <= remaining else block[:remaining]
        if snippet:
            selected.append((index, snippet))
            used += len(snippet) + 2
    selected.sort(key=lambda item: item[0])
    return "\n\n".join(block for _, block in selected)[:limit]


def analyze_tdr_with_rag(candidate_text, ranked_references):
    """Arbitre un TDR avec Qwen3 à partir des références récupérées par le moteur."""
    references = []
    for item in ranked_references[:RAG_REFERENCE_TOP_K]:
        document = item.get("_document")
        if document is None:
            continue
        source_text = (getattr(document, "extracted_text", None)
                       or getattr(document, "cleaned_text", None) or "").strip()
        references.append({
            "source": item.get("source") or item.get("title") or f"Référence {item.get('id')}",
            "title": item.get("title") or "",
            "retrieval_scores": {
                "lexical": item.get("tfidf_score"),
                "semantic": item.get("semantic_score"),
                "hybrid": item.get("hybrid_score"),
            },
            "text": _reference_excerpt(candidate_text, source_text, MAX_REFERENCE_CHARS),
        })

    if not references:
        raise OllamaRAGError("Aucune référence indexée n'a pu être transmise à Qwen3.")

    payload = {
        "tdr_candidat": (candidate_text or "")[:MAX_CANDIDATE_CHARS],
        "references_recuperees": references,
        "consigne": (
            "Analyse le TDR candidat par rapport à chaque référence. Les références "
            "sont des résultats de recherche, pas nécessairement pertinentes. "
            "Choisis le cas métier le mieux justifié et cite les éléments probants."
        ),
    }
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "think": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "options": {"temperature": 0.1},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except requests.Timeout as exc:
        raise OllamaRAGError(
            f"Ollama n'a pas répondu dans le délai de {OLLAMA_TIMEOUT:g} secondes. "
            "Le prompt a été limité pour tenir dans le contexte du modèle; vérifiez aussi la charge CPU."
        ) from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "inconnu"
        body = exc.response.text[:500] if exc.response is not None else str(exc)
        raise OllamaRAGError(
            f"Ollama a renvoyé HTTP {status}: {body}"
        ) from exc
    except requests.RequestException as exc:
        raise OllamaRAGError(
            f"Ollama est inaccessible à {OLLAMA_BASE_URL}. Vérifiez qu'Ollama est démarré "
            f"et que le modèle « {OLLAMA_MODEL} » est installé."
        ) from exc
    except ValueError as exc:
        raise OllamaRAGError("Ollama a renvoyé une réponse HTTP non valide.") from exc

    message = data.get("message") or {}
    result = _response_json(message.get("content", ""))
    allowed_cases = {
        "CAS_1_CORRESPONDANCE_FORTE",
        "CAS_2_CORRESPONDANCE_PARTIELLE",
        "CAS_3_MEME_OBJECTIF_ZONE_DIFFERENTE",
        "CAS_4_AUCUNE_ETUDE_PERTINENTE",
    }
    allowed_decisions = {"SIMILAIRE", "SIMILARITÉ PARTIELLE", "DIFFERENT", "À EXAMINER"}
    if result.get("case_id") not in allowed_cases or result.get("decision") not in allowed_decisions:
        raise OllamaRAGError("La réponse de Qwen3 ne respecte pas le format de décision attendu.")

    # Le case_id est la classification métier canonique : on évite qu'une
    # réponse JSON incohérente associe un cas et une décision contradictoires.
    case_decisions = {
        "CAS_1_CORRESPONDANCE_FORTE": "SIMILAIRE",
        "CAS_2_CORRESPONDANCE_PARTIELLE": "SIMILARITÉ PARTIELLE",
        "CAS_3_MEME_OBJECTIF_ZONE_DIFFERENTE": "DIFFERENT",
        "CAS_4_AUCUNE_ETUDE_PERTINENTE": "DIFFERENT",
    }
    result["decision"] = case_decisions[result["case_id"]]

    for key in ("common_points", "differences", "evidence", "missing_information"):
        if not isinstance(result.get(key), list):
            result[key] = []
    for key in ("case_title", "explanation", "recommendation", "confidence"):
        if not isinstance(result.get(key), str):
            result[key] = ""
    valid_sources = {ref["source"] for ref in references}
    matched_reference = result.get("matched_reference")
    if not isinstance(matched_reference, str) or matched_reference not in valid_sources:
        matched_reference = ""
    result["matched_reference"] = matched_reference

    # Ne conserver que les preuves rattachées à une source effectivement fournie à Qwen.
    validated_evidence = []
    for item in result["evidence"]:
        if not isinstance(item, dict):
            continue
        source_name = item.get("source")
        if source_name not in valid_sources:
            continue
        validated_evidence.append({
            "source": source_name,
            "point": str(item.get("point") or "")[:300],
            "justification": str(item.get("justification") or "")[:700],
        })
    result["evidence"] = validated_evidence
    result["model"] = OLLAMA_MODEL
    result["provider"] = "ollama"
    result["references_used"] = [ref["source"] for ref in references]
    result["retrieval_top_k"] = len(references)
    return result

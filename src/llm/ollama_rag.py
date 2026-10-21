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
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
RAG_REFERENCE_TOP_K = max(1, int(os.getenv("TDR_RAG_REFERENCE_TOP_K", "5")))
MAX_CANDIDATE_CHARS = max(3000, int(os.getenv("TDR_RAG_CANDIDATE_MAX_CHARS", "18000")))
MAX_REFERENCE_CHARS = max(1500, int(os.getenv("TDR_RAG_REFERENCE_MAX_CHARS", "7000")))


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
            "text": source_text[:MAX_REFERENCE_CHARS],
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
            f"Ollama n'a pas répondu dans le délai de {OLLAMA_TIMEOUT:g} secondes."
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

    for key in ("common_points", "differences", "evidence", "missing_information"):
        if not isinstance(result.get(key), list):
            result[key] = []
    for key in ("case_title", "explanation", "recommendation", "confidence"):
        if not isinstance(result.get(key), str):
            result[key] = ""
    result["model"] = OLLAMA_MODEL
    result["provider"] = "ollama"
    result["references_used"] = [ref["source"] for ref in references]
    result["retrieval_top_k"] = len(references)
    return result

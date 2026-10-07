"""Juge IA local pour comparer deux études avec Ollama."""

import json
import os
import re
from typing import Any

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_ENABLED = os.getenv("OLLAMA_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}

_SYSTEM_PROMPT = """Tu es le juge intelligent de TDRDOC-SCAN.
Tu compares deux documents TDR pour déterminer s'ils portent réellement sur la même étude.
Tu dois LIRE L'ENSEMBLE des informations fournies avant de conclure.

REGLE CENTRALE :
Une étude n'est pas un doublon simplement parce qu'elle partage le même secteur,
la même zone, une méthodologie similaire, des formulations administratives
similaires ou les mêmes types de livrables. L'OBJET REEL DE L'ETUDE est prioritaire.

Distingue toujours :
- secteur/domaine ;
- objet réel de l'étude ;
- zone géographique ;
- objectifs et résultats ;
- activités/composantes ;
- population/bénéficiaires ;
- méthodologie ;
- éléments administratifs génériques.

Une reformulation importante ne signifie PAS que les études sont différentes.
Reconnais une même étude reformulée lorsque son identité métier reste la même.

A l'inverse, si les documents sont dans le même secteur et la même zone mais
traitent de problèmes ou objets d'étude différents, le verdict doit être
DIFFERENT même si leur structure et leur méthodologie se ressemblent beaucoup.

Les scores numériques sont seulement des indices. Ton jugement porte sur le sens global.

Réponds UNIQUEMENT avec un objet JSON valide :
{
  "verdict": "SIMILAIRE|DIFFERENT|A_EXAMINER",
  "confidence": "FORTE|MOYENNE|FAIBLE",
  "objet_a": "...",
  "objet_b": "...",
  "points_communs": ["..."],
  "differences_decisives": ["..."],
  "raison": "explication courte en français"
}
"""

def _trim(text: str, limit: int = 14000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...[contenu central tronqué]...\n" + text[-half:]

def _extract_json(raw: str) -> dict[str, Any] | None:
    raw = (raw or "").strip()
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None

def judge_documents(
    candidate_text: str,
    source_text: str,
    contextual_analysis: dict[str, Any] | None = None,
    technical_evidence: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Demande à Ollama un jugement global. None si Ollama est indisponible."""
    if not OLLAMA_ENABLED:
        return None

    context = contextual_analysis or {}
    evidence = technical_evidence or {}
    prompt = f"""{_SYSTEM_PROMPT}

INDICES TECHNIQUES (ils ne remplacent pas ton jugement) :
- similarité TF-IDF : {evidence.get("tfidf_score")}
- similarité sémantique : {evidence.get("semantic_score")}
- score hybride : {evidence.get("hybrid_score")}
- couverture des passages : {evidence.get("coverage")}
- analyse contextuelle : {json.dumps(context, ensure_ascii=False)}

DOCUMENT A — DOCUMENT CONTROLE :
{_trim(candidate_text)}

DOCUMENT B — DOCUMENT DE REFERENCE :
{_trim(source_text)}

Lis les deux documents ensemble. Identifie leur objet réel et décide s'il
s'agit réellement de la même étude. Les similitudes génériques ne doivent pas
suffire à conclure SIMILAIRE.
"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        result = _extract_json(payload.get("message", {}).get("content", ""))
        if not result:
            return None

        verdict = str(result.get("verdict", "")).upper().replace("À", "A")
        if verdict not in {"SIMILAIRE", "DIFFERENT", "A_EXAMINER"}:
            return None

        confidence = str(result.get("confidence", "MOYENNE")).upper()
        if confidence not in {"FORTE", "MOYENNE", "FAIBLE"}:
            confidence = "MOYENNE"

        result["verdict"] = verdict
        result["confidence"] = confidence
        return result
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return None

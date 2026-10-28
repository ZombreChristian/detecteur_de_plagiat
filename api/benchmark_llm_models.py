"""Compare le temps de réponse de plusieurs modèles via le serveur OpenAI-compatible.

Depuis la racine du dépôt :
    python api/benchmark_llm_models.py
    python api/benchmark_llm_models.py qwen3.5-9b qwen3.6-35b

La clé reste dans frontend/.env et n'est jamais affichée.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / "frontend" / ".env")

BASE_URL = (
    os.getenv("TDR_LLM_BASE_URL")
    or "https://meyer-variable-halo-labeled.trycloudflare.com/api"
).strip().rstrip("/")
API_KEY = os.getenv("TDR_LLM_API_KEY", "").strip()

# Liste utilisée si aucun nom de modèle n'est passé en argument.
DEFAULT_MODELS = ["qwen3.5-9b", "qwen3.6-35b"]
MODELS = sys.argv[1:] or DEFAULT_MODELS

if not API_KEY:
    sys.exit(
        "ERREUR : TDR_LLM_API_KEY est absente. "
        "Ajoute la clé dans frontend/.env sans la publier sur GitHub."
    )

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=180, max_retries=0)
prompt = (
    "Réponds directement, sans raisonnement détaillé, en une phrase courte. "
    "Écris uniquement : Test réussi."
)

print(f"Serveur : {BASE_URL}")
print(f"Modèles à tester : {', '.join(MODELS)}")
print("Chaque modèle reçoit exactement la même demande.\n")

results = []
for model in MODELS:
    print("=" * 64)
    print(f"Modèle : {model}")
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        elapsed = time.perf_counter() - start
        if not response.choices:
            print(f"ÉCHEC après {elapsed:.2f} s : aucune réponse (choices vide).")
            results.append((model, elapsed, "ÉCHEC : choices vide"))
            continue

        choice = response.choices[0]
        message = choice.message
        answer = (getattr(message, "content", None) or "").strip()
        finish = getattr(choice, "finish_reason", None)
        if answer:
            print(f"Temps de réponse : {elapsed:.2f} secondes")
            print(f"Fin de génération : {finish}")
            print(f"Réponse : {answer[:300]}")
            results.append((model, elapsed, "OK"))
        else:
            reasoning = (getattr(message, "reasoning_content", None) or "").strip()
            detail = "content vide"
            if reasoning:
                detail += ", reasoning_content présent mais non final"
            print(f"ÉCHEC après {elapsed:.2f} s : {detail}.")
            print(f"Fin de génération : {finish}")
            results.append((model, elapsed, "ÉCHEC : " + detail))
    except Exception as exc:
        elapsed = time.perf_counter() - start
        # Ne jamais afficher la clé API.
        print(f"ÉCHEC après {elapsed:.2f} s : {type(exc).__name__}: {exc}")
        results.append((model, elapsed, f"ÉCHEC : {type(exc).__name__}"))

print("\n" + "=" * 64)
print("RÉCAPITULATIF")
print(f"{'Modèle':<24} {'Temps (s)':>12}  Résultat")
print("-" * 64)
for model, elapsed, status in sorted(results, key=lambda row: row[1]):
    print(f"{model:<24} {elapsed:>12.2f}  {status}")
print("\nLe temps mesuré inclut l'attente réseau et la génération du fournisseur.")

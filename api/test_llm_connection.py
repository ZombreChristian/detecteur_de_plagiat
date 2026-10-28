"""Test manuel du serveur LLM distant depuis le dossier api.

Usage (depuis la racine du dépôt) :
    python api/test_llm_connection.py

La clé doit rester dans frontend/.env (TDR_LLM_API_KEY), jamais dans Git.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / "frontend" / ".env")

base_url = (
    os.getenv("TDR_LLM_BASE_URL")
    or "https://meyer-variable-halo-labeled.trycloudflare.com/api"
).strip().rstrip("/")
api_key = os.getenv("TDR_LLM_API_KEY", "").strip()
model = (os.getenv("TDR_LLM_MODEL") or "qwen3.6-35b").strip()

if not api_key:
    sys.exit(
        "ERREUR : TDR_LLM_API_KEY est absente. "
        "Ajoute la clé dans frontend/.env sans la publier sur GitHub."
    )

print(f"URL : {base_url}")
print(f"Modèle demandé : {model}")
print("Test de connexion en cours...")

try:
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120, max_retries=0)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Bonjour"}],
        temperature=0,
        max_tokens=256,
    )
    print(f"Modèle retourné : {getattr(response, 'model', None)}")
    if not response.choices:
        sys.exit("ÉCHEC : le serveur n'a renvoyé aucun choix.")
    choice = response.choices[0]
    message = choice.message
    content = (getattr(message, "content", None) or "").strip()
    print(f"Fin de génération : {getattr(choice, 'finish_reason', None)}")
    if content:
        print("Réponse :")
        print(content)
    else:
        reasoning = (getattr(message, "reasoning_content", None) or "").strip()
        print("ÉCHEC : la réponse finale (content) est vide.")
        if reasoning:
            print("Le serveur a renvoyé un reasoning_content, mais celui-ci n'est pas la réponse finale.")
        sys.exit(2)
except Exception as exc:
    # Ne jamais afficher la clé API dans les diagnostics.
    print(f"ÉCHEC de l'appel au fournisseur : {type(exc).__name__}: {exc}")
    sys.exit(1)

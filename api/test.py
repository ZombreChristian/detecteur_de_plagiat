"""Test de connexion à un endpoint compatible avec l'API OpenAI.

Configuration par variables d'environnement :
  MODEL_API_BASE_URL (facultatif)
  MODEL_API_KEY (obligatoire)
  MODEL_NAME (facultatif)
"""
import os
import sys

from openai import OpenAI

BASE_URL = os.getenv(
    "MODEL_API_BASE_URL",
    "https://meyer-variable-halo-labeled.trycloudflare.com/api",
)
API_KEY = os.getenv("MODEL_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.6-35b")

if not API_KEY:
    sys.exit(
        "Clé API manquante. Définis MODEL_API_KEY (ou OPENAI_API_KEY) "
        "dans l'environnement avant de lancer ce script."
    )

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Bonjour ! Réponds brièvement en français."}],
    )
    print(response.choices[0].message.content or "(Réponse vide)")
except Exception as exc:
    sys.exit(f"Échec de l'appel à l'API ({type(exc).__name__}) : {exc}")

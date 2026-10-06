"""Gestionnaire du modèle de similarité.

Le modèle est téléchargé une seule fois lors de la première utilisation puis
sauvegardé dans le dossier local modeles/. Les exécutions suivantes chargent
uniquement la copie locale.
"""
from functools import lru_cache
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = os.getenv(
    "SIMILARITY_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
_MODEL_DIR_VALUE = os.getenv(
    "SIMILARITY_MODEL_DIR",
    "modeles/paraphrase-multilingual-MiniLM-L12-v2",
)
MODEL_DIR = Path(_MODEL_DIR_VALUE)
if not MODEL_DIR.is_absolute():
    MODEL_DIR = ROOT / MODEL_DIR


def _is_local_model_ready() -> bool:
    return MODEL_DIR.is_dir() and (MODEL_DIR / "modules.json").is_file()


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Retourne une instance unique du modèle pour tout le processus."""
    if _is_local_model_ready():
        return SentenceTransformer(str(MODEL_DIR), local_files_only=True)

    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN") or None

    model = SentenceTransformer(
        MODEL_NAME,
        token=hf_token,
        cache_folder=os.getenv("HF_HOME") or None,
    )
    model.save(str(MODEL_DIR))
    return model


def get_model_info() -> dict:
    return {
        "name": MODEL_NAME,
        "local_path": str(MODEL_DIR),
        "local_ready": _is_local_model_ready(),
    }

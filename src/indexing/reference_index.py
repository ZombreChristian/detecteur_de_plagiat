"""Construction et lecture de l'index documentaire PostgreSQL.

Le corpus de référence est indexé une fois :
DOCX -> extraction -> nettoyage -> embedding -> PostgreSQL.

Les fichiers envoyés par les utilisateurs restent des candidats temporaires et
ne sont jamais ajoutés automatiquement à la base de référence.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from django.core.management import call_command
from django.db import transaction
from django.utils import timezone

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend"
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "frontend.settings")

from src.extraction.extract_docx import extract_docx
from src.preprocessing.clean_text import clean_text
from src.similarity.model_manager import MODEL_NAME, get_model


CORPUS = {
    "duplicate": (ROOT / "donnees" / "TDR", "TDR"),
    "plagiarism": (ROOT / "donnees" / "Rapport d'etude", "RAPPORT"),
}


def setup_django() -> None:
    import django

    django.setup()


def ensure_database_schema() -> None:
    """Applique les migrations Django nécessaires pour le registre/index."""
    setup_django()
    call_command("migrate", interactive=False, verbosity=0)


def _source_documents() -> list[tuple[Path, str]]:
    documents: list[tuple[Path, str]] = []
    for folder, document_type in CORPUS.values():
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.docx")):
            documents.append((path, document_type))
    return documents


def _relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def index_reference_corpus(force: bool = False) -> dict[str, int]:
    """Indexe les documents de référence et évite tout recalcul inutile."""
    ensure_database_schema()

    from detector.models import StudyDocument

    sources = _source_documents()
    if not sources:
        raise RuntimeError(
            "Aucun document DOCX trouvé dans donnees/TDR ou donnees/Rapport d'etude."
        )

    model = get_model()
    pending: list[dict[str, Any]] = []
    skipped = 0

    for path, document_type in sources:
        extracted = extract_docx(path)
        cleaned = clean_text(extracted)
        if not cleaned:
            continue

        file_path = _relative_path(path)
        content_hash = _hash_text(cleaned)
        existing = StudyDocument.objects.filter(file_path=file_path).first()

        if (
            existing
            and existing.is_reference
            and existing.embedding
            and existing.content_hash == content_hash
            and existing.embedding_model == MODEL_NAME
            and not force
        ):
            skipped += 1
            continue

        pending.append(
            {
                "path": path,
                "file_path": file_path,
                "document_type": document_type,
                "title": path.stem,
                "extracted_text": extracted,
                "cleaned_text": cleaned,
                "content_hash": content_hash,
                "existing": existing,
            }
        )

    if not pending:
        return {
            "created_or_updated": 0,
            "skipped": skipped,
            "total": len(sources),
        }

    embeddings = model.encode(
        [item["cleaned_text"] for item in pending],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    with transaction.atomic():
        for item, vector in zip(pending, embeddings):
            document = item["existing"]
            if document is None:
                document = StudyDocument(
                    title=item["title"],
                    document_type=item["document_type"],
                    file_path=item["file_path"],
                )

            document.title = item["title"]
            document.document_type = item["document_type"]
            document.file_path = item["file_path"]
            document.is_reference = True
            document.extracted_text = item["extracted_text"]
            document.cleaned_text = item["cleaned_text"]
            document.embedding = [float(value) for value in np.asarray(vector, dtype=np.float32)]
            document.embedding_model = MODEL_NAME
            document.content_hash = item["content_hash"]
            document.indexed_at = timezone.now()
            document.status = "Référence indexée"
            document.save()

    return {
        "created_or_updated": len(pending),
        "skipped": skipped,
        "total": len(sources),
    }


def ensure_reference_index() -> dict[str, int]:
    """Construit automatiquement l'index au premier démarrage uniquement."""
    ensure_database_schema()

    from detector.models import StudyDocument

    references = list(
        StudyDocument.objects.filter(is_reference=True).only("id", "embedding")
    )
    if references and all(bool(item.embedding) for item in references):
        return {
            "created_or_updated": 0,
            "skipped": len(references),
            "total": len(references),
        }

    return index_reference_corpus(force=False)


def load_reference_documents(kind: str):
    """Charge uniquement les références correspondant au mode d'analyse."""
    setup_django()

    from detector.models import StudyDocument

    expected_type = CORPUS[kind][1]
    rows = list(
        StudyDocument.objects.filter(
            is_reference=True,
            document_type=expected_type,
        ).order_by("id")
    )
    rows = [row for row in rows if row.cleaned_text and row.embedding]
    if not rows:
        raise RuntimeError(
            f"Aucun document de référence indexé pour le mode « {kind} »."
        )

    embeddings = np.asarray([row.embedding for row in rows], dtype=np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    return rows, embeddings

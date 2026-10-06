from pathlib import Path
import sys

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = (
        "Prépare le corpus de référence : extraction, nettoyage et embeddings "
        "stockés dans PostgreSQL."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Recalcule les embeddings même lorsqu'un document est déjà indexé.",
        )

    def handle(self, *args, **options):
        root = Path(settings.BASE_DIR).parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        try:
            from src.indexing.reference_index import index_reference_corpus

            stats = index_reference_corpus(force=options["force"])
        except Exception as exc:
            raise CommandError(f"Préparation de l'index impossible : {exc}") from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Index documentaire prêt : "
                f"{stats['created_or_updated']} créé(s)/mis à jour, "
                f"{stats['skipped']} déjà indexé(s), "
                f"{stats['total']} document(s) détecté(s)."
            )
        )

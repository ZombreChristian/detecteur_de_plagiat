from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Accorde le rôle Administrateur à un utilisateur TDRDOC-SCAN."

    def add_arguments(self, parser):
        parser.add_argument("username", help="Nom d'utilisateur à promouvoir")

    def handle(self, *args, **options):
        User = get_user_model()
        username = options["username"]
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist as exc:
            raise CommandError(f"Utilisateur introuvable : {username}") from exc

        user.is_staff = True
        user.is_active = True
        user.save(update_fields=["is_staff", "is_active"])

        self.stdout.write(
            self.style.SUCCESS(
                f"Le compte « {user.username} » est maintenant Administrateur et actif."
            )
        )

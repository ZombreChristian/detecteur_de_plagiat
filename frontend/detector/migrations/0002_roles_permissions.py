from django.db import migrations

def create_default_roles(apps, schema_editor):
    Group = apps.get_model("auth", "Group")
    Permission = apps.get_model("auth", "Permission")
    ct = apps.get_model("contenttypes", "ContentType").objects.get(app_label="detector", model="studydocument")
    perms = {p.codename: p for p in Permission.objects.filter(content_type=ct)}
    user_group, _ = Group.objects.get_or_create(name="Utilisateur")
    admin_group, _ = Group.objects.get_or_create(name="Administrateur")
    user_codes = {"can_verify_tdr", "can_analyze_report", "can_view_history", "can_export_results"}
    user_group.permissions.set([perms[c] for c in user_codes if c in perms])
    admin_group.permissions.set(list(perms.values()))

class Migration(migrations.Migration):
    dependencies = [("detector", "0001_initial")]
    operations = [
        migrations.AlterModelOptions(
            name="studydocument",
            options={"permissions": [
                ("can_verify_tdr", "Vérifier un TDR / détecter un doublon"),
                ("can_analyze_report", "Analyser un rapport / détecter le plagiat"),
                ("can_view_history", "Consulter l'historique des analyses"),
                ("can_export_results", "Exporter les résultats PDF / Excel"),
                ("can_manage_registry", "Gérer le registre des études"),
                ("can_manage_users", "Gérer les utilisateurs, rôles et permissions"),
                ("can_manage_settings", "Gérer les paramètres de l'application"),
            ]},
        ),
        migrations.RunPython(create_default_roles, migrations.RunPython.noop),
    ]

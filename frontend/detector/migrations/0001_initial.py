from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion

class Migration(migrations.Migration):
    initial = True
    dependencies = [migrations.swappable_dependency(settings.AUTH_USER_MODEL)]
    operations = [
        migrations.CreateModel(name="StudyDocument", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("title", models.CharField(max_length=500)), ("object", models.TextField(blank=True)),
            ("geographic_scope", models.CharField(blank=True, max_length=255)), ("sector", models.CharField(blank=True, max_length=255)),
            ("expected_results", models.TextField(blank=True)), ("commissioning_entity", models.CharField(blank=True, max_length=255)),
            ("year", models.PositiveIntegerField(blank=True, null=True)), ("budget", models.DecimalField(blank=True, decimal_places=2, max_digits=15, null=True)),
            ("status", models.CharField(blank=True, max_length=100)), ("document_type", models.CharField(choices=[("TDR", "TDR"), ("RAPPORT", "Rapport d'étude")], max_length=20)),
            ("file_path", models.CharField(blank=True, max_length=1000)), ("created_at", models.DateTimeField(auto_now_add=True)), ("updated_at", models.DateTimeField(auto_now=True))]),
        migrations.CreateModel(name="WhitelistedPassage", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")), ("text", models.TextField(unique=True)),
            ("reason", models.CharField(blank=True, max_length=500)), ("active", models.BooleanField(default=True)), ("created_at", models.DateTimeField(auto_now_add=True))]),
        migrations.CreateModel(name="Analysis", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")), ("document_name", models.CharField(max_length=255)),
            ("mode", models.CharField(choices=[("plagiarism", "Plagiat"), ("duplicate", "Doublon")], max_length=20)), ("decision", models.CharField(blank=True, max_length=100)),
            ("hybrid_score", models.FloatField(blank=True, null=True)), ("semantic_score", models.FloatField(blank=True, null=True)), ("tfidf_score", models.FloatField(blank=True, null=True)),
            ("novelty_score", models.FloatField(blank=True, null=True)), ("result_json", models.JSONField(default=dict)), ("status", models.CharField(default="success", max_length=30)),
            ("error_message", models.TextField(blank=True)), ("duration_ms", models.PositiveIntegerField(blank=True, null=True)), ("created_at", models.DateTimeField(auto_now_add=True)),
            ("document", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to="detector.studydocument")),
            ("user", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
        ]),
    ]

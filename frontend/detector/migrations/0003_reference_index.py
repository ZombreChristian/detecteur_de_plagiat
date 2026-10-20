from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("detector", "0002_roles_permissions"),
    ]

    operations = [
        migrations.AddField(
            model_name="studydocument",
            name="is_reference",
            field=models.BooleanField(db_index=True, default=False),
        ),
        migrations.AddField(
            model_name="studydocument",
            name="extracted_text",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="studydocument",
            name="cleaned_text",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="studydocument",
            name="embedding",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="studydocument",
            name="embedding_model",
            field=models.CharField(blank=True, max_length=255),
        ),
        migrations.AddField(
            model_name="studydocument",
            name="content_hash",
            field=models.CharField(blank=True, db_index=True, max_length=64),
        ),
        migrations.AddField(
            model_name="studydocument",
            name="indexed_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]

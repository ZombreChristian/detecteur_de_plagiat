from django.conf import settings
from django.db import models

class StudyDocument(models.Model):
    TYPE_CHOICES = [("TDR", "TDR"), ("RAPPORT", "Rapport d'étude")]
    title = models.CharField(max_length=500)
    object = models.TextField(blank=True)
    geographic_scope = models.CharField(max_length=255, blank=True)
    sector = models.CharField(max_length=255, blank=True)
    expected_results = models.TextField(blank=True)
    commissioning_entity = models.CharField(max_length=255, blank=True)
    year = models.PositiveIntegerField(null=True, blank=True)
    budget = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    status = models.CharField(max_length=100, blank=True)
    document_type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    file_path = models.CharField(max_length=1000, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

class WhitelistedPassage(models.Model):
    text = models.TextField(unique=True)
    reason = models.CharField(max_length=500, blank=True)
    active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

class Analysis(models.Model):
    MODE_CHOICES = [("plagiarism", "Plagiat"), ("duplicate", "Doublon")]
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    document = models.ForeignKey(StudyDocument, null=True, blank=True, on_delete=models.SET_NULL)
    document_name = models.CharField(max_length=255)
    mode = models.CharField(max_length=20, choices=MODE_CHOICES)
    decision = models.CharField(max_length=100, blank=True)
    hybrid_score = models.FloatField(null=True, blank=True)
    semantic_score = models.FloatField(null=True, blank=True)
    tfidf_score = models.FloatField(null=True, blank=True)
    novelty_score = models.FloatField(null=True, blank=True)
    result_json = models.JSONField(default=dict)
    status = models.CharField(max_length=30, default="success")
    error_message = models.TextField(blank=True)
    duration_ms = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

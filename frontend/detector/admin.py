from django.contrib import admin
from .models import Analysis, StudyDocument, WhitelistedPassage

@admin.register(StudyDocument)
class StudyDocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "document_type", "sector", "year", "status", "created_at")
    search_fields = ("title", "sector", "commissioning_entity")
    list_filter = ("document_type", "sector", "status")

@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = ("document_name", "mode", "decision", "hybrid_score", "novelty_score", "created_at")
    search_fields = ("document_name", "decision")
    list_filter = ("mode", "decision", "status")
    readonly_fields = ("created_at",)

@admin.register(WhitelistedPassage)
class WhitelistedPassageAdmin(admin.ModelAdmin):
    list_display = ("text", "reason", "active", "created_at")
    list_filter = ("active",)

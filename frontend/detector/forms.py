from django import forms
from django.contrib.auth.forms import AuthenticationForm
from .models import StudyDocument, WhitelistedPassage


class LoginForm(AuthenticationForm):
    username = forms.CharField(
        label="Nom d'utilisateur",
        widget=forms.TextInput(attrs={"class": "input", "placeholder": "Nom d'utilisateur", "autofocus": True}),
    )
    password = forms.CharField(
        label="Mot de passe",
        widget=forms.PasswordInput(attrs={"class": "input", "placeholder": "Mot de passe"}),
    )


class ReceptionForm(forms.Form):
    tdr_document = forms.FileField(label="TDR accepté", required=False)
    report_document = forms.FileField(label="Rapport d'étude", required=False)
    mode = forms.ChoiceField(
        label="Type de réception",
        choices=[
            ("tdr", "Vérifier un TDR avant validation"),
            ("report", "Réceptionner le rapport associé à un TDR"),
        ],
    )


class StudyDocumentForm(forms.ModelForm):
    class Meta:
        model = StudyDocument
        fields = (
            "title",
            "document_type",
            "object",
            "geographic_scope",
            "sector",
            "expected_results",
            "commissioning_entity",
            "year",
            "budget",
            "status",
            "file_path",
        )
        widgets = {
            "title": forms.TextInput(attrs={"placeholder": "Titre de l'étude"}),
            "object": forms.Textarea(attrs={"rows": 3, "placeholder": "Objet de l'étude"}),
            "geographic_scope": forms.TextInput(attrs={"placeholder": "Périmètre géographique"}),
            "sector": forms.TextInput(attrs={"placeholder": "Secteur"}),
            "expected_results": forms.Textarea(attrs={"rows": 3, "placeholder": "Résultats attendus"}),
            "commissioning_entity": forms.TextInput(attrs={"placeholder": "Structure commanditaire"}),
            "year": forms.NumberInput(attrs={"min": 2000}),
            "budget": forms.NumberInput(attrs={"step": "0.01", "min": "0"}),
            "status": forms.TextInput(attrs={"placeholder": "Statut"}),
            "file_path": forms.TextInput(attrs={"placeholder": "Chemin du fichier, si disponible"}),
        }


class WhitelistedPassageForm(forms.ModelForm):
    class Meta:
        model = WhitelistedPassage
        fields = ("text", "reason", "active")
        widgets = {
            "text": forms.Textarea(attrs={"rows": 4, "placeholder": "Passage autorisé à ignorer lors du contrôle"}),
            "reason": forms.TextInput(attrs={"placeholder": "Motif de l'autorisation"}),
        }

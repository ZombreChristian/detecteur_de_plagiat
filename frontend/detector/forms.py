from django import forms
from django.contrib.auth.forms import AuthenticationForm


class LoginForm(AuthenticationForm):
    username = forms.CharField(label="Nom d'utilisateur", widget=forms.TextInput(attrs={"class": "input", "placeholder": "Nom d'utilisateur", "autofocus": True}))
    password = forms.CharField(label="Mot de passe", widget=forms.PasswordInput(attrs={"class": "input", "placeholder": "Mot de passe"}))


class ReceptionForm(forms.Form):
    tdr_document = forms.FileField(label="TDR accepté", required=False)
    report_document = forms.FileField(label="Rapport d'étude", required=False)
    mode = forms.ChoiceField(label="Type de réception", choices=[("tdr", "Vérifier un TDR avant validation"), ("report", "Réceptionner le rapport associé à un TDR")])

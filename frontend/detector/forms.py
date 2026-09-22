from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm, PasswordChangeForm
from .models import StudyDocument, WhitelistedPassage

User = get_user_model()


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


class SignUpForm(UserCreationForm):
    email = forms.EmailField(label="Adresse e-mail", required=True)
    first_name = forms.CharField(label="Prénom", required=False)
    last_name = forms.CharField(label="Nom", required=False)

    class Meta:
        model = User
        fields = ("username", "first_name", "last_name", "email", "password1", "password2")


class UserPasswordChangeForm(PasswordChangeForm):
    old_password = forms.CharField(
        label="Ancien mot de passe",
        widget=forms.PasswordInput(attrs={"autocomplete": "current-password"}),
    )
    new_password1 = forms.CharField(
        label="Nouveau mot de passe",
        widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}),
    )
    new_password2 = forms.CharField(
        label="Confirmer le nouveau mot de passe",
        widget=forms.PasswordInput(attrs={"autocomplete": "new-password"}),
    )


class PasswordRecoveryForm(forms.Form):
    last_name = forms.CharField(label="Nom", max_length=150)
    first_name = forms.CharField(label="Prénom", max_length=150)
    email = forms.EmailField(label="Adresse e-mail")

    def clean(self):
        cleaned = super().clean()
        last_name = (cleaned.get("last_name") or "").strip()
        first_name = (cleaned.get("first_name") or "").strip()
        email = (cleaned.get("email") or "").strip()

        if not last_name or not first_name or not email:
            return cleaned

        user = User.objects.filter(
            last_name__iexact=last_name,
            first_name__iexact=first_name,
            email__iexact=email,
            is_active=True,
        ).first()

        if not user:
            raise forms.ValidationError(
                "Aucun compte actif ne correspond à ces nom, prénom et adresse e-mail."
            )

        self.user = user
        return cleaned


class AdminUserCreateForm(UserCreationForm):
    ROLE_CHOICES = (
        ("user", "Utilisateur"),
        ("admin", "Administrateur"),
    )
    first_name = forms.CharField(label="Prénom", required=False)
    last_name = forms.CharField(label="Nom", required=False)
    email = forms.EmailField(label="Adresse e-mail", required=False)
    role = forms.ChoiceField(label="Rôle", choices=ROLE_CHOICES, initial="user")
    is_active = forms.BooleanField(label="Compte actif", required=False, initial=True)

    class Meta:
        model = User
        fields = (
            "username",
            "first_name",
            "last_name",
            "email",
            "role",
            "is_active",
            "password1",
            "password2",
        )

    def save(self, commit=True):
        user = super().save(commit=False)
        user.is_staff = self.cleaned_data["role"] == "admin"
        user.is_active = self.cleaned_data["is_active"]
        if commit:
            user.save()
        return user


class AdminUserUpdateForm(forms.ModelForm):
    ROLE_CHOICES = (
        ("user", "Utilisateur"),
        ("admin", "Administrateur"),
    )
    role = forms.ChoiceField(label="Rôle", choices=ROLE_CHOICES)
    is_active = forms.BooleanField(label="Compte actif", required=False)

    class Meta:
        model = User
        fields = ("username", "first_name", "last_name", "email", "role", "is_active")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial["role"] = "admin" if self.instance.is_staff else "user"

    def save(self, commit=True):
        user = super().save(commit=False)
        user.is_staff = self.cleaned_data["role"] == "admin"
        user.is_active = self.cleaned_data["is_active"]
        if commit:
            user.save()
        return user


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

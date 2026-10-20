import io
from io import BytesIO
import re
import os
import time
import requests
from pathlib import Path
from django.core.files.storage import FileSystemStorage
from docx import Document as DocxDocument
from docx.shared import RGBColor
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, get_user_model, update_session_auth_hash
from django.contrib.auth.models import Group, Permission
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.urls import reverse
from django.contrib.auth.tokens import default_token_generator
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode
from django.db.models import Q
from django.http import FileResponse, HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from openpyxl import Workbook
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from .forms import LoginForm, SignUpForm, AdminUserCreateForm, AdminUserUpdateForm, PasswordRecoveryForm, UserPasswordChangeForm
from .models import Analysis, StudyDocument


def login_view(request):
    if request.user.is_authenticated:
        return redirect("detector:dashboard")
    form = LoginForm(request, data=request.POST or None)
    if request.method == "POST" and form.is_valid():
        login(request, form.get_user())
        return redirect("detector:dashboard")
    return render(request, "login.html", {"form": form})


def password_reset_request(request):
    if request.user.is_authenticated:
        return redirect("detector:dashboard")
    form = PasswordRecoveryForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.user
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        token = default_token_generator.make_token(user)
        reset_path = reverse("detector:password_reset_confirm", kwargs={"uidb64": uid, "token": token})
        base_url = getattr(settings, "PUBLIC_APP_URL", "").rstrip("/")
        reset_url = f"{base_url}{reset_path}" if base_url else f"{'https' if request.is_secure() else 'http'}://{request.get_host()}{reset_path}"
        context = {"email": user.email, "user": user, "domain": request.get_host(), "site_name": "TDRDOC-SCAN", "protocol": "https" if request.is_secure() else "http", "uid": uid, "token": token, "reset_url": reset_url}
        subject = render_to_string("password_reset_subject.txt", context).strip()
        message = render_to_string("password_reset_email.txt", context)
        required_email_settings = (getattr(settings, "EMAIL_HOST", ""), getattr(settings, "EMAIL_HOST_USER", ""), getattr(settings, "EMAIL_HOST_PASSWORD", ""), getattr(settings, "DEFAULT_FROM_EMAIL", ""))
        if not all(required_email_settings):
            form.add_error(None, "L'envoi e-mail n'est pas configuré. Renseignez EMAIL_HOST, EMAIL_HOST_USER, EMAIL_HOST_PASSWORD et DEFAULT_FROM_EMAIL dans le fichier .env.")
        else:
            try:
                send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [user.email], fail_silently=False)
            except Exception as exc:
                form.add_error(None, f"Échec de l'envoi SMTP : {exc}" if settings.DEBUG else "Le message de récupération n'a pas pu être envoyé. Vérifiez la configuration e-mail du serveur.")
            else:
                return redirect("detector:password_reset_done")
    return render(request, "password_reset.html", {"form": form})


def password_reset_done(request): return render(request, "password_reset_done.html")

def password_reset_confirm(request, uidb64, token):
    from django.contrib.auth.views import PasswordResetConfirmView
    view = PasswordResetConfirmView.as_view(template_name="password_reset_confirm.html", success_url="/mot-de-passe-reinitialise/")
    return view(request, uidb64=uidb64, token=token)

def password_reset_complete(request): return render(request, "password_reset_complete.html")

def signup_view(request):
    if request.user.is_authenticated: return redirect("detector:dashboard")
    form = SignUpForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        form.save(); messages.success(request, "Votre compte a été créé. Vous pouvez maintenant vous connecter."); return redirect("detector:login")
    return render(request, "signup.html", {"form": form})


@login_required
def account_settings(request):
    form = UserPasswordChangeForm(request.user, request.POST or None)
    if request.method == "POST" and form.is_valid():
        form.save(); update_session_auth_hash(request, request.user); messages.success(request, "Votre mot de passe a été modifié avec succès."); return redirect("detector:account_settings")
    return render(request, "account_settings.html", {"form": form})


def logout_view(request): logout(request); return redirect("detector:login")


def _functional_permissions():
    return list(Permission.objects.filter(content_type__app_label="detector", codename__startswith="can_").order_by("codename"))


def _ensure_administrator_role(user):
    """Synchronise un compte administrateur avec le rôle Administrateur et toutes les permissions fonctionnelles."""
    if not user or not user.is_authenticated or not (user.is_superuser or user.is_staff):
        return
    role, _ = Group.objects.get_or_create(name="Administrateur")
    permissions = _functional_permissions()
    if permissions and role.permissions.count() != len(permissions):
        role.permissions.set(permissions)
    if not user.groups.filter(pk=role.pk).exists():
        user.groups.add(role)


def _has_permission(user, codename):
    if not user or not user.is_authenticated:
        return False
    if user.is_superuser:
        return True
    _ensure_administrator_role(user)
    return user.has_perm(f"detector.{codename}")


def _require_permission(request, codename, message):
    if _has_permission(request.user, codename): return True
    messages.error(request, message)
    return False


@login_required
def administration_utilisateur_supprimer(request, pk):
    if not request.user.is_staff:
        messages.error(request, "Accès réservé aux administrateurs.")
        return redirect("detector:dashboard")
    User = get_user_model(); target = get_object_or_404(User, pk=pk)
    if target.pk == request.user.pk:
        messages.error(request, "Vous ne pouvez pas supprimer votre propre compte."); return redirect("detector:administration")
    if request.method == "POST":
        target.is_active = False; target.save(update_fields=["is_active"]); messages.success(request, f"Le compte « {target.username} » a été désactivé."); return redirect("detector:administration_users")
    return render(request, "administration_utilisateur.html", {"target_user": target, "profile_form": AdminUserUpdateForm(instance=target), "groups": Group.objects.order_by("name"), "current_group": target.groups.first(), "delete_confirmation": True})


def _get_uploaded_docx(request):
    candidates = ("document", "tdr_document", "report_document", "file")
    for field_name in candidates:
        uploaded = request.FILES.get(field_name)
        if uploaded and uploaded.name: return uploaded
    for uploaded in request.FILES.values():
        if uploaded and uploaded.name: return uploaded
    return None


def _validate_document(uploaded): return bool(uploaded and uploaded.name and Path(uploaded.name).suffix.lower() in {".docx", ".pdf"})
def _document_content_type(uploaded): return "application/pdf" if Path(uploaded.name).suffix.lower() == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@login_required
def dashboard(request):
    _ensure_administrator_role(request.user)
    base = Analysis.objects.filter(user=request.user).order_by("-created_at")
    q = request.GET.get("q", "").strip(); decision = request.GET.get("decision", "all"); mode = request.GET.get("mode", "all")
    if q: base = base.filter(Q(document_name__icontains=q) | Q(result_json__best_source__icontains=q))
    if decision in ("SIMILAIRE", "DIFFERENT", "À EXAMINER"): base = base.filter(decision=decision)
    if mode in ("duplicate", "plagiarism"): base = base.filter(mode=mode)
    paginator = Paginator(base, 10); page_obj = paginator.get_page(request.GET.get("page", 1))
    total = Analysis.objects.filter(user=request.user).count(); similar = Analysis.objects.filter(user=request.user, decision="SIMILAIRE").count(); different = Analysis.objects.filter(user=request.user, decision="DIFFERENT").count(); to_review = Analysis.objects.filter(user=request.user, decision="À EXAMINER").count(); tdr_checks = Analysis.objects.filter(user=request.user, mode="duplicate").count(); report_checks = Analysis.objects.filter(user=request.user, mode="plagiarism").count()
    registry_total = StudyDocument.objects.count(); registry_tdr = StudyDocument.objects.filter(document_type="TDR").count(); registry_rapport = StudyDocument.objects.filter(document_type="RAPPORT").count(); has_admin = get_user_model().objects.filter(is_staff=True, is_active=True).exists()
    return render(request, "dashboard.html", {"recent": page_obj.object_list, "page_obj": page_obj, "total_filtered": paginator.count, "q": q, "selected_decision": decision, "selected_mode": mode, "total": total, "similar": similar, "different": different, "to_review": to_review, "tdr_checks": tdr_checks, "report_checks": report_checks, "registry_total": registry_total, "registry_tdr": registry_tdr, "registry_rapport": registry_rapport, "has_admin": has_admin})



def _save_uploaded_document(uploaded):
    media_root = Path(settings.MEDIA_ROOT) / "documents"; media_root.mkdir(parents=True, exist_ok=True); storage = FileSystemStorage(location=str(media_root)); uploaded.seek(0); saved_name = storage.save(uploaded.name, uploaded); uploaded.seek(0); return f"documents/{saved_name}"

def _document_url(relative_path): return f"{settings.MEDIA_URL.rstrip('/')}/{relative_path.lstrip('/')}" if relative_path else ""
def _safe_reference_path(document):
    if not document or not document.file_path: return None
    root = Path(settings.BASE_DIR).resolve().parent; candidate = (root / document.file_path).resolve()
    try: candidate.relative_to(root)
    except ValueError: return None
    return candidate if candidate.is_file() else None


@login_required
def open_reference_document(request, pk):
    document = get_object_or_404(StudyDocument, pk=pk, is_reference=True); path = _safe_reference_path(document)
    if not path: messages.error(request, "Le fichier de référence n'est plus disponible."); return redirect("detector:dashboard")
    content_type = "application/pdf" if path.suffix.lower() == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return FileResponse(path.open("rb"), as_attachment=False, filename=path.name, content_type=content_type)


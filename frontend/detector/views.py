import io
import time
import requests
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import render, redirect
from openpyxl import Workbook
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from .forms import LoginForm
from .models import Analysis, StudyDocument


def login_view(request):
    if request.user.is_authenticated:
        return redirect("detector:dashboard")
    form = LoginForm(request, data=request.POST or None)
    if request.method == "POST" and form.is_valid():
        login(request, form.get_user())
        return redirect("detector:dashboard")
    return render(request, "login.html", {"form": form})


def logout_view(request):
    logout(request)
    return redirect("detector:login")


@login_required
def dashboard(request):
    recent = Analysis.objects.filter(user=request.user).order_by("-created_at")[:8]
    total = Analysis.objects.filter(user=request.user).count()
    similar = Analysis.objects.filter(user=request.user, decision="SIMILAIRE").count()
    different = Analysis.objects.filter(user=request.user, decision="DIFFERENT").count()
    to_review = Analysis.objects.filter(user=request.user, decision="A EXAMINER").count()
    tdr_checks = Analysis.objects.filter(user=request.user, mode="duplicate").count()
    report_checks = Analysis.objects.filter(user=request.user, mode="plagiarism").count()
    return render(request, "dashboard.html", {"recent": recent, "total": total, "similar": similar, "different": different, "to_review": to_review, "tdr_checks": tdr_checks, "report_checks": report_checks})


@login_required
def reception(request):
    """Parcours métier : vérifier le TDR, puis autoriser la réception du rapport associé."""
    if request.method == "GET":
        return render(request, "reception.html")
    mode = request.POST.get("mode", "tdr")
    uploaded = request.FILES.get("document")
    if not uploaded or not uploaded.name.lower().endswith(".docx"):
        messages.error(request, "Veuillez sélectionner un document DOCX valide.")
        return redirect("detector:reception")
    endpoint = "/api/detect/duplicate" if mode == "tdr" else "/api/detect/plagiarism"
    try:
        response = requests.post(f"{settings.FASTAPI_URL}{endpoint}", files={"file": (uploaded.name, uploaded.file, uploaded.content_type)}, timeout=600)
        response.raise_for_status()
        result = response.json()
        analysis = Analysis.objects.create(user=request.user, document_name=uploaded.name, mode="duplicate" if mode == "tdr" else "plagiarism", decision=result.get("decision", ""), hybrid_score=result.get("hybrid_score"), semantic_score=result.get("semantic_score"), tfidf_score=result.get("tfidf_score"), novelty_score=result.get("novelty_score"), result_json=result, duration_ms=result.get("duration_ms"))
        request.session["last_result"] = result
        request.session["last_analysis_id"] = analysis.id
        request.session["last_document_name"] = uploaded.name
        if mode == "tdr" and result.get("decision") == "DIFFERENT":
            request.session["validated_tdr_name"] = uploaded.name
            messages.success(request, "TDR accepté : aucune similarité suffisante n'a été trouvée. Le rapport associé peut maintenant être réceptionné.")
            return redirect("detector:reception")
        if mode == "tdr" and result.get("decision") == "SIMILAIRE":
            messages.error(request, "TDR non retenu : un document similaire existe déjà dans la base.")
        else:
            messages.info(request, "Rapport analysé. Consultez le détail des résultats.")
        return redirect("detector:resultats")
    except requests.RequestException as exc:
        messages.error(request, f"Le moteur d'analyse est indisponible : {exc}")
    return redirect("detector:reception")


@login_required
def analyser(request):
    if request.method != "POST":
        return render(request, "analyser.html")
    uploaded = request.FILES.get("document")
    mode = request.POST.get("mode", "plagiarism")
    if mode not in ("plagiarism", "duplicate"):
        messages.error(request, "Mode d'analyse invalide.")
        return redirect("detector:analyser")
    if not uploaded or not uploaded.name.lower().endswith(".docx"):
        messages.error(request, "Veuillez sélectionner un document DOCX valide.")
        return redirect("detector:analyser")
    endpoint = "/api/detect/duplicate" if mode == "duplicate" else "/api/detect/plagiarism"
    start = time.perf_counter()
    try:
        response = requests.post(f"{settings.FASTAPI_URL}{endpoint}", files={"file": (uploaded.name, uploaded.file, uploaded.content_type)}, timeout=600)
        response.raise_for_status()
        result = response.json()
        duration = round((time.perf_counter() - start) * 1000)
        result["duration_ms"] = duration
        Analysis.objects.create(user=request.user, document_name=uploaded.name, mode=mode, decision=result.get("decision", ""), hybrid_score=result.get("hybrid_score"), semantic_score=result.get("semantic_score"), tfidf_score=result.get("tfidf_score"), novelty_score=result.get("novelty_score"), result_json=result, duration_ms=duration)
        request.session["last_result"] = result
        request.session["last_document_name"] = uploaded.name
        return redirect("detector:resultats")
    except requests.RequestException as exc:
        messages.error(request, f"Le moteur FastAPI est indisponible : {exc}")
    except Exception as exc:
        messages.error(request, f"Erreur pendant l'analyse : {exc}")
    return render(request, "analyser.html")


@login_required
def resultats(request):
    result = request.session.get("last_result")
    recent = Analysis.objects.filter(user=request.user).order_by("-created_at")[:20]
    return render(request, "resultats.html", {"result": result, "recent": recent, "document_name": request.session.get("last_document_name", "")})


@login_required
def export_excel(request):
    analysis = Analysis.objects.filter(user=request.user).order_by("-created_at").first()
    if not analysis:
        return HttpResponse("Aucune analyse à exporter.", status=404)
    result = analysis.result_json or {}
    wb = Workbook(); ws = wb.active; ws.title = "Analyse"
    for row in [["Détecteur documentaire", "DOCSEC"], ["Document", analysis.document_name], ["Mode", analysis.get_mode_display()], ["Décision", analysis.decision], ["Score TF-IDF", analysis.tfidf_score], ["Score sémantique", analysis.semantic_score], ["Score hybride", analysis.hybrid_score], ["Score de nouveauté", analysis.novelty_score], ["Durée (ms)", analysis.duration_ms], ["Meilleure source", result.get("best_source", "")]]: ws.append(row)
    response = HttpResponse(content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"); response["Content-Disposition"] = 'attachment; filename="rapport_analyse.xlsx"'; wb.save(response); return response


@login_required
def export_pdf(request):
    analysis = Analysis.objects.filter(user=request.user).order_by("-created_at").first()
    if not analysis: return HttpResponse("Aucune analyse à exporter.", status=404)
    result = analysis.result_json or {}; buffer = io.BytesIO(); doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36); styles = getSampleStyleSheet()
    story = [Paragraph("Rapport d'analyse documentaire — DOCSEC", styles["Title"]), Spacer(1, 12)]
    data = [["Document", analysis.document_name], ["Type", analysis.get_mode_display()], ["Décision", analysis.decision], ["Score TF-IDF", str(analysis.tfidf_score)], ["Score sémantique", str(analysis.semantic_score)], ["Score hybride", str(analysis.hybrid_score)], ["Nouveauté", str(analysis.novelty_score)], ["Meilleure source", str(result.get("best_source", ""))]]
    table = Table(data, colWidths=[150, 350]); table.setStyle(TableStyle([("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eef2f7")), ("GRID", (0,0), (-1,-1), .5, colors.grey), ("VALIGN", (0,0), (-1,-1), "TOP"), ("PADDING", (0,0), (-1,-1), 7)])); story.append(table); story.append(Spacer(1, 18)); story.append(Paragraph("Sources proches", styles["Heading2"]))
    for source in result.get("sources", [])[:10]: story.extend([Paragraph(f"{source.get('source','')} — score {source.get('hybrid_score','')}", styles["BodyText"]), Spacer(1, 5)])
    doc.build(story); response = HttpResponse(buffer.getvalue(), content_type="application/pdf"); response["Content-Disposition"] = 'attachment; filename="rapport_analyse.pdf"'; return response

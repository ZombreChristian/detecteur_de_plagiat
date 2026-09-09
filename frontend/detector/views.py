import io
import time
import requests
from django.conf import settings
from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import render, redirect
from openpyxl import Workbook
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from .models import Analysis


def dashboard(request):
    recent = Analysis.objects.order_by("-created_at")[:8]
    total = Analysis.objects.count()
    similar = Analysis.objects.filter(decision="SIMILAIRE").count()
    return render(request, "dashboard.html", {"recent": recent, "total": total, "similar": similar})


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
        response = requests.post(
            f"{settings.FASTAPI_URL}{endpoint}",
            files={"file": (uploaded.name, uploaded.file, uploaded.content_type)},
            timeout=600,
        )
        response.raise_for_status()
        result = response.json()
        duration = round((time.perf_counter() - start) * 1000)
        result["duration_ms"] = duration
        Analysis.objects.create(
            user=request.user if request.user.is_authenticated else None,
            document_name=uploaded.name,
            mode=mode,
            decision=result.get("decision", ""),
            hybrid_score=result.get("hybrid_score"),
            semantic_score=result.get("semantic_score"),
            tfidf_score=result.get("tfidf_score"),
            novelty_score=result.get("novelty_score"),
            result_json=result,
            duration_ms=duration,
        )
        request.session["last_result"] = result
        request.session["last_document_name"] = uploaded.name
        return redirect("detector:resultats")
    except requests.RequestException as exc:
        messages.error(request, f"Le moteur FastAPI est indisponible : {exc}")
    except Exception as exc:
        messages.error(request, f"Erreur pendant l'analyse : {exc}")
    return render(request, "analyser.html")


def resultats(request):
    result = request.session.get("last_result")
    recent = Analysis.objects.order_by("-created_at")[:20]
    return render(request, "resultats.html", {
        "result": result,
        "recent": recent,
        "document_name": request.session.get("last_document_name", ""),
    })


def export_excel(request):
    analysis = Analysis.objects.order_by("-created_at").first()
    if not analysis:
        return HttpResponse("Aucune analyse à exporter.", status=404)
    result = analysis.result_json or {}
    wb = Workbook()
    ws = wb.active
    ws.title = "Analyse"
    rows = [
        ["Détecteur documentaire", "DOCSEC"],
        ["Document", analysis.document_name],
        ["Mode", analysis.get_mode_display()],
        ["Décision", analysis.decision],
        ["Score TF-IDF", analysis.tfidf_score],
        ["Score sémantique", analysis.semantic_score],
        ["Score hybride", analysis.hybrid_score],
        ["Score de nouveauté", analysis.novelty_score],
        ["Durée (ms)", analysis.duration_ms],
        ["Meilleure source", result.get("best_source", "")],
    ]
    for row in rows:
        ws.append(row)
    response = HttpResponse(content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response["Content-Disposition"] = 'attachment; filename="rapport_analyse.xlsx"'
    wb.save(response)
    return response


def export_pdf(request):
    analysis = Analysis.objects.order_by("-created_at").first()
    if not analysis:
        return HttpResponse("Aucune analyse à exporter.", status=404)
    result = analysis.result_json or {}
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [Paragraph("Rapport d'analyse documentaire — DOCSEC", styles["Title"]), Spacer(1, 12)]
    data = [
        ["Document", analysis.document_name], ["Type", analysis.get_mode_display()],
        ["Décision", analysis.decision], ["Score TF-IDF", str(analysis.tfidf_score)],
        ["Score sémantique", str(analysis.semantic_score)], ["Score hybride", str(analysis.hybrid_score)],
        ["Nouveauté", str(analysis.novelty_score)], ["Meilleure source", str(result.get("best_source", ""))],
    ]
    table = Table(data, colWidths=[150, 350])
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eef2f7")), ("GRID", (0,0), (-1,-1), .5, colors.grey), ("VALIGN", (0,0), (-1,-1), "TOP"), ("PADDING", (0,0), (-1,-1), 7)]))
    story.append(table)
    story.append(Spacer(1, 18))
    story.append(Paragraph("Sources proches", styles["Heading2"]))
    for source in result.get("sources", [])[:10]:
        story.append(Paragraph(f"{source.get('source','')} — score {source.get('hybrid_score','')}", styles["BodyText"]))
        story.append(Spacer(1, 5))
    doc.build(story)
    response = HttpResponse(buffer.getvalue(), content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="rapport_analyse.pdf"'
    return response

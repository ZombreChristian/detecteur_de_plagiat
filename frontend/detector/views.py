import time
import requests
from django.conf import settings
from django.contrib import messages
from django.shortcuts import render, redirect
from .models import Analysis


def dashboard(request):
    recent = Analysis.objects.order_by("-created_at")[:8]
    return render(request, "dashboard.html", {"recent": recent})


def analyser(request):
    if request.method != "POST":
        return render(request, "analyser.html")
    uploaded = request.FILES.get("document")
    mode = request.POST.get("mode", "plagiarism")
    if not uploaded:
        messages.error(request, "Veuillez sélectionner un document DOCX.")
        return redirect("detector:analyser")
    endpoint = "/api/detect/duplicate" if mode == "duplicate" else "/api/detect/plagiarism"
    start = time.perf_counter()
    try:
        response = requests.post(f"{settings.FASTAPI_URL}{endpoint}", files={"file": (uploaded.name, uploaded.file, uploaded.content_type)}, timeout=600)
        response.raise_for_status(); result = response.json()
        Analysis.objects.create(user=request.user if request.user.is_authenticated else None, document_name=uploaded.name, mode=mode, decision=result.get("decision",""), hybrid_score=result.get("hybrid_score"), semantic_score=result.get("semantic_score"), tfidf_score=result.get("tfidf_score"), novelty_score=result.get("novelty_score"), result_json=result, duration_ms=round((time.perf_counter()-start)*1000))
        request.session["last_result"] = result
        return redirect("detector:resultats")
    except requests.RequestException:
        messages.error(request, "Le moteur d'analyse est indisponible. Lancez FastAPI sur le port 8000.")
    except Exception as exc:
        messages.error(request, f"Erreur pendant l'analyse : {exc}")
    return render(request, "analyser.html")


def resultats(request):
    return render(request, "resultats.html", {"result": request.session.get("last_result"), "recent": Analysis.objects.order_by("-created_at")[:20]})

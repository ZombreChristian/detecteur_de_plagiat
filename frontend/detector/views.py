from pathlib import Path
import requests
from django.conf import settings
from django.shortcuts import render


def dashboard(request):
    return render(request, "dashboard.html")


def analyse(request):
    context = {"result": None, "error": None}
    if request.method != "POST":
        return render(request, "dashboard.html", context)

    uploaded = request.FILES.get("document")
    mode = request.POST.get("mode", "plagiarism")
    if not uploaded:
        context["error"] = "Veuillez sélectionner un document DOCX."
        return render(request, "dashboard.html", context)

    endpoint = "/api/detect/duplicate" if mode == "duplicate" else "/api/detect/plagiarism"
    try:
        response = requests.post(
            f"{settings.FASTAPI_URL}{endpoint}",
            files={"file": (uploaded.name, uploaded.file, uploaded.content_type)},
            timeout=300,
        )
        response.raise_for_status()
        context["result"] = response.json()
    except requests.RequestException as exc:
        context["error"] = (
            "Impossible de contacter le moteur FastAPI. "
            "Vérifiez que l'API est démarrée sur le port 8000."
        )
    return render(request, "dashboard.html", context)

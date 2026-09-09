from django.urls import path
from . import views

app_name = "detector"
urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("analyser/", views.analyser, name="analyser"),
    path("resultats/", views.resultats, name="resultats"),
    path("export/excel/", views.export_excel, name="export_excel"),
    path("export/pdf/", views.export_pdf, name="export_pdf"),
]

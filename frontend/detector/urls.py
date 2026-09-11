from django.urls import path
from . import views

app_name = "detector"
urlpatterns = [
    path("connexion/", views.login_view, name="login"),
    path("deconnexion/", views.logout_view, name="logout"),
    path("", views.dashboard, name="dashboard"),
    path("reception/", views.reception, name="reception"),
    path("analyser/", views.analyser, name="analyser"),
    path("resultats/", views.resultats, name="resultats"),
    path("analyse/<int:pk>/", views.analysis_detail, name="analysis_detail"),
    path("export/excel/", views.export_excel, name="export_excel"),
    path("export/pdf/", views.export_pdf, name="export_pdf"),
]

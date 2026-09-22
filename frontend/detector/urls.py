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
    path("administration/", views.administration, name="administration"),
    path("analyse/<int:pk>/", views.analysis_detail, name="analysis_detail"),
    path("analyse/<int:pk>/rapport/", views.upload_report_for_tdr, name="upload_report_for_tdr"),
    path("export/excel/", views.export_excel, name="export_excel"),
    path("export/pdf/", views.export_pdf, name="export_pdf"),
]

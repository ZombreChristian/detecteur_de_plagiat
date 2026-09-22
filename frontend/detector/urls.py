from django.urls import path
from . import views

app_name = "detector"
urlpatterns = [
    path("connexion/", views.login_view, name="login"),
    path("mot-de-passe-oublie/", views.password_reset_request, name="password_reset"),
    path("mot-de-passe-oublie/envoye/", views.password_reset_done, name="password_reset_done"),
    path("mot-de-passe-reinitialiser/<uidb64>/<token>/", views.password_reset_confirm, name="password_reset_confirm"),
    path("mot-de-passe-reinitialise/", views.password_reset_complete, name="password_reset_complete"),
    path("deconnexion/", views.logout_view, name="logout"),
    path("", views.dashboard, name="dashboard"),
    path("reception/", views.reception, name="reception"),
    path("analyser/", views.analyser, name="analyser"),
    path("resultats/", views.resultats, name="resultats"),
    path("administration/", views.administration, name="administration"),
    path("administration/recuperation/", views.administration_recovery, name="administration_recovery"),
    path("administration/utilisateur/<int:pk>/", views.administration_utilisateur, name="administration_utilisateur"),
    path("administration/utilisateur/<int:pk>/supprimer/", views.administration_utilisateur_supprimer, name="administration_utilisateur_supprimer"),
    path("inscription/", views.signup_view, name="signup"),
    path("analyse/<int:pk>/", views.analysis_detail, name="analysis_detail"),
    path("analyse/<int:pk>/rapport/", views.upload_report_for_tdr, name="upload_report_for_tdr"),
    path("export/excel/", views.export_excel, name="export_excel"),
    path("export/pdf/", views.export_pdf, name="export_pdf"),
]

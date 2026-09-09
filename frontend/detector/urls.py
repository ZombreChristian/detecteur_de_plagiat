from django.urls import path
from . import views
app_name = "detector"
urlpatterns = [path("", views.dashboard, name="dashboard"), path("analyser/", views.analyser, name="analyser"), path("resultats/", views.resultats, name="resultats")]

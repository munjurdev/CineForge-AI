"""
URL configuration for the system_monitor app.

CineForge AI — Local AI Video Generation Platform.

System diagnostics, model setup, and settings pages.
"""
from __future__ import annotations

from django.urls import path

from . import views

app_name = "system_monitor"

urlpatterns = [
    path("diagnostics/", views.diagnostics, name="diagnostics"),
    path("model-setup/", views.model_setup, name="model_setup"),
    path("settings/", views.settings_page, name="settings"),
]

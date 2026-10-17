"""
URL configuration for the core app.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.urls import path

from . import views

app_name = "core"

urlpatterns = [
    path("", views.home, name="home"),
    path("dashboard/", views.dashboard, name="dashboard"),
]

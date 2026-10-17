"""
App configuration for the core app.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"
    verbose_name = "CineForge Core"

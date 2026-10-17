"""
App configuration for the generator app.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.apps import AppConfig


class GeneratorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "generator"
    verbose_name = "CineForge Generator"

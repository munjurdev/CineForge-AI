"""
App configuration for the library app.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.apps import AppConfig


class LibraryConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "library"
    verbose_name = "CineForge Library"

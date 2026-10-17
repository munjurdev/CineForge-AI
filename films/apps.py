from __future__ import annotations

from django.apps import AppConfig


class FilmsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "films"
    verbose_name = "CineForge Short-Film Studio"

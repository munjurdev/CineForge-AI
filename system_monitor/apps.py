"""App configuration for the system_monitor app.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.apps import AppConfig


class SystemMonitorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "system_monitor"
    verbose_name = "CineForge System Monitor"

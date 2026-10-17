"""
CineForge AI — Local AI Video Generation Platform.

Celery app is exposed here so that ``celery -A cineforge ...`` works.
"""
from __future__ import annotations

from .celery import app as celery_app

__all__ = ["celery_app"]

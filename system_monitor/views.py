"""
System_monitor app views — diagnostics, model setup, and settings.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

import logging

from django.conf import settings as django_settings
from django.shortcuts import render

from .services import detect_system_info

logger = logging.getLogger("cineforge.system_monitor")


def diagnostics(request):
    """Render the system diagnostics page."""
    info = detect_system_info()
    return render(request, "system_monitor/diagnostics.html", {
        "page": "diagnostics",
        "info": info,
    })


def model_setup(request):
    """Render the model setup page."""
    info = detect_system_info()
    return render(request, "system_monitor/model_setup.html", {
        "page": "model_setup",
        "info": info,
        "default_adapter": django_settings.INFERENCE_DEFAULT_ADAPTER,
        "comfyui_base_url": django_settings.COMFYUI_BASE_URL,
    })


def settings_page(request):
    """Render the read-only configuration overview."""
    config = {
        "debug": django_settings.DEBUG,
        "allowed_hosts": ", ".join(django_settings.ALLOWED_HOSTS),
        "time_zone": django_settings.TIME_ZONE,
        "media_root": django_settings.MEDIA_ROOT,
        "static_root": django_settings.STATIC_ROOT,
        "max_upload_size_mb": django_settings.MAX_UPLOAD_SIZE_MB,
        "allowed_image_extensions": ", ".join(
            django_settings.ALLOWED_IMAGE_EXTENSIONS
        ),
        "default_adapter": django_settings.INFERENCE_DEFAULT_ADAPTER,
        "comfyui_base_url": django_settings.COMFYUI_BASE_URL,
        "comfyui_timeout": django_settings.COMFYUI_API_TIMEOUT,
        "comfyui_workflow_path": (
            getattr(django_settings, "COMFYUI_WORKFLOW_PATH", "") or "Not configured"
        ),
        "celery_broker": django_settings.CELERY_BROKER_URL,
        "celery_backend": django_settings.CELERY_RESULT_BACKEND,
        "celery_eager": getattr(django_settings, "CELERY_TASK_ALWAYS_EAGER", False),
        "system_monitor_enabled": django_settings.ENABLE_SYSTEM_MONITOR,
    }
    return render(request, "system_monitor/settings.html", {
        "page": "settings",
        "config": config,
    })

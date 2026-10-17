"""
Global context processor for CineForge UI chrome and feature flags.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.conf import settings


def app_settings(request):
    """Inject a small set of site-wide settings into every template context."""
    return {
        "app_name": "CineForge AI",
        "debug": settings.DEBUG,
        "media_url": settings.MEDIA_URL,
        "static_url": settings.STATIC_URL,
        "max_upload_size_mb": getattr(settings, "MAX_UPLOAD_SIZE_MB", 50),
        "allowed_image_extensions": getattr(
            settings, "ALLOWED_IMAGE_EXTENSIONS", []
        ),
        "ffmpeg_available": _ffmpeg_available(),
        "enable_system_monitor": getattr(
            settings, "ENABLE_SYSTEM_MONITOR", True
        ),
    }


def _ffmpeg_available() -> bool:
    """Return True when the ffmpeg binary configured in settings can be found."""
    import shutil

    from django.conf import settings

    bin_name = getattr(settings, "FFMPEG_BIN", "ffmpeg")
    path = shutil.which(bin_name)
    return path is not None

"""
URL configuration for the CineForge AI project.

CineForge AI — Local AI Video Generation Platform.

The root URLconf includes the core, generator, library, films, and
system_monitor apps. Admin and static/media serving are configured for
local development only.
"""
from __future__ import annotations

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("core.urls")),
    path("generate/", include("generator.urls")),
    path("library/", include("library.urls")),
    path("films/", include("films.urls")),
    path("system/", include("system_monitor.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

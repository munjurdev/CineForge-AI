"""
URL configuration for the library app.

CineForge AI — Local AI Video Generation Platform.

Provides the video gallery and per-video detail/download/delete views.
"""
from __future__ import annotations

from django.urls import path

from . import views

app_name = "library"

urlpatterns = [
    path("", views.video_list, name="video_list"),
    path("video/<uuid:pk>/", views.video_detail, name="video_detail"),
    path("video/<uuid:pk>/download/", views.video_download, name="video_download"),
    path("video/<uuid:pk>/delete/", views.video_delete, name="video_delete"),
]

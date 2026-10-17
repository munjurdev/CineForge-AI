"""
URL configuration for the films app.

CineForge AI — Local AI Video Generation Platform.

Short-film studio: create projects, manage scenes, and assemble the final
movie with FFmpeg.
"""
from __future__ import annotations

from django.urls import path

from . import views

app_name = "films"

urlpatterns = [
    path("", views.film_list, name="film_list"),
    path("new/", views.film_new, name="film_new"),
    path("characters/", views.character_list, name="character_list"),
    path("<uuid:pk>/", views.film_detail, name="film_detail"),
    path("<uuid:pk>/edit/", views.film_edit, name="film_edit"),
    path("<uuid:pk>/assemble/", views.film_assemble, name="film_assemble"),
    path("<uuid:pk>/download/", views.film_download, name="film_download"),
    path("<uuid:pk>/scene/add/", views.scene_add, name="scene_add"),
    path(
        "<uuid:pk>/scene/<uuid:scene_pk>/generate/",
        views.scene_generate,
        name="scene_generate",
    ),
    path(
        "<uuid:project_pk>/scene/<uuid:scene_pk>/move/<str:direction>/",
        views.scene_move,
        name="scene_move",
    ),
    path(
        "<uuid:project_pk>/scene/<uuid:scene_pk>/delete/",
        views.scene_delete,
        name="scene_delete",
    ),
]

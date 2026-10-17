"""
Admin registration for the films app.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.contrib import admin

from .models import CharacterReference, FilmProject, FilmScene


class FilmSceneInline(admin.TabularInline):
    model = FilmScene
    extra = 0
    fields = ["order", "title", "status", "duration_seconds", "output_file", "transition_in", "transition_out"]
    readonly_fields = ["status"]


@admin.register(FilmProject)
class FilmProjectAdmin(admin.ModelAdmin):
    list_display = ["title", "created", "assembled"]
    list_filter = ["created"]
    search_fields = ["title", "description"]
    inlines = [FilmSceneInline]

    @admin.display(description="Assembled")
    def assembled(self, obj: FilmProject) -> bool:
        return bool(obj.assembled_video)


@admin.register(FilmScene)
class FilmSceneAdmin(admin.ModelAdmin):
    list_display = ["order", "title", "project", "status", "duration_seconds"]
    list_filter = ["status", "project"]
    search_fields = ["title", "prompt"]
    readonly_fields = ["status"]


@admin.register(CharacterReference)
class CharacterReferenceAdmin(admin.ModelAdmin):
    list_display = ["name", "created"]
    search_fields = ["name", "description"]

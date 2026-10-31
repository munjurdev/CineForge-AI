"""
Admin configuration for the generator app.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.contrib import admin

from .models import GenerationJob, GeneratedVideo


@admin.register(GenerationJob)
class GenerationJobAdmin(admin.ModelAdmin):
    list_display = [
        "id_short",
        "status",
        "engine",
        "model_name",
        "resolution",
        "duration",
        "created",
        "updated",
    ]
    list_filter = ["status", "engine", "model_name"]
    search_fields = ["prompt", "model_name", "id"]
    readonly_fields = [
        "id",
        "created",
        "updated",
        "task_id",
        "status",
    ]
    fieldsets = (
        ("Job info", {
            "fields": ("id", "status", "engine", "model_name", "model_variant",
                       "created", "updated", "task_id"),
        }),
        ("Prompt", {
            "fields": ("prompt", "negative_prompt"),
        }),
        ("Parameters", {
            "fields": (
                "duration_seconds", "width", "height", "fps",
                "aspect_ratio", "seed", "guidance_scale", "steps",
            ),
        }),
        ("Result", {
            "fields": (
                "progress", "message", "output_file", "error_type",
                "log_snippet",
            ),
        }),
    )

    def id_short(self, obj: GenerationJob) -> str:
        return f"{obj.id.hex[:8]}…"

    id_short.short_description = "Job id"

    def get_form(self, request, obj=None, **kwargs):
        """Give the admin a real engine <select>.

        ``GenerationJob.engine`` no longer declares ``choices`` (the provider
        registry owns that list, so backends can be added without a
        migration), which would otherwise render as free text here.
        """
        from django import forms

        from .providers import engine_choices

        form = super().get_form(request, obj, **kwargs)
        field = form.base_fields.get("engine")
        if field is not None:
            form.base_fields["engine"] = forms.ChoiceField(
                choices=engine_choices(),
                required=field.required,
                label="Engine",
            )
        return form

    def resolution(self, obj: GenerationJob) -> str:
        return f"{obj.width}×{obj.height}"

    resolution.short_description = "Resolution"

    def duration(self, obj: GenerationJob) -> str:
        return f"{obj.duration_seconds}s"


@admin.register(GeneratedVideo)
class GeneratedVideoAdmin(admin.ModelAdmin):
    list_display = [
        "id_short",
        "title",
        "duration_display",
        "resolution",
        "size_mb",
        "created",
    ]
    list_filter = ["created"]
    search_fields = ["title", "prompt_summary", "id"]
    readonly_fields = [
        "id", "created", "updated", "size_mb_display",
    ]
    fieldsets = (
        ("Video", {
            "fields": ("id", "title", "file", "thumbnail", "created", "updated"),
        }),
        ("Metadata", {
            "fields": (
                "duration_seconds", "width", "height", "fps",
                "codec", "size_bytes",
            ),
        }),
        ("Context", {
            "fields": ("job", "prompt_summary"),
        }),
    )

    def id_short(self, obj: GeneratedVideo) -> str:
        return f"{obj.id.hex[:8]}…"

    id_short.short_description = "Video id"

    def duration_display(self, obj: GeneratedVideo) -> str:
        return obj.duration_display

    duration_display.short_description = "Duration"

    def resolution(self, obj: GeneratedVideo) -> str:
        if obj.width and obj.height:
            return f"{obj.width}×{obj.height}"
        return "—"

    resolution.short_description = "Resolution"

    def size_mb_display(self, obj: GeneratedVideo) -> str:
        return obj.natural_size()

    size_mb_display.short_description = "Size"

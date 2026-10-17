from __future__ import annotations

import uuid

from django.db import models

from generator.models import GeneratedVideo


def _upload_to_films(instance: models.Model, filename: str) -> str:
    from django.conf import settings
    from pathlib import Path
    import uuid
    return str(settings.MEDIA_OUTPUTS_DIR / "films" / uuid.uuid4().hex / Path(filename).name)


class FilmProject(models.Model):
    """A short-film project composed of multiple scenes."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    title = models.CharField(max_length=255, help_text="Film title.")
    description = models.TextField(blank=True, help_text="Optional synopsis.")
    project_note = models.TextField(
        blank=True,
        help_text="Notes about character consistency, continuity, and style.",
    )

    assembled_video = models.FileField(
        upload_to=_upload_to_films,
        blank=True,
        null=True,
        help_text="Final assembled movie, when created.",
    )
    assembly_error = models.TextField(
        blank=True,
        help_text="Error message from the last assembly attempt.",
    )

    class Meta:
        ordering = ["-created"]

    def __str__(self) -> str:
        return f"Film {self.id.hex[:8]} — {self.title}"

    def can_assemble(self) -> bool:
        from generator.services import probe_video_file
        from pathlib import Path

        scenes = self.scenes.filter(
            status=FilmScene.Status.SUCCESS,
            output_file__isnull=False,
        )
        viable = 0
        for scene in scenes:
            if scene.output_file and scene.output_file.path:
                p = Path(scene.output_file.path)
                if p.exists():
                    meta = probe_video_file(p)
                    if meta:
                        viable += 1
        return viable >= 2

    def assembly_output_path(self):
        from pathlib import Path
        if self.assembled_video:
            return Path(self.assembled_video.path)
        return None


class FilmScene(models.Model):
    """A single scene within a film project."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    project = models.ForeignKey(
        FilmProject,
        on_delete=models.CASCADE,
        related_name="scenes",
    )
    order = models.PositiveIntegerField(default=0, help_text="Scene order in the film.")
    title = models.CharField(
        max_length=255,
        blank=True,
        help_text="Scene title or slug.",
    )
    prompt = models.TextField(help_text="Scene generation prompt.")
    negative_prompt = models.TextField(
        blank=True,
        default="",
        help_text="Optional negative prompt.",
    )
    character_reference = models.ForeignKey(
        "films.CharacterReference",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="scenes",
        help_text="Reusable character profile, where supported by the model.",
    )
    source_video = models.ForeignKey(
        GeneratedVideo,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="used_in_scenes",
        help_text="Generated clip used as the scene's video.",
    )
    job = models.ForeignKey(
        "generator.GenerationJob",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="film_scenes",
        help_text="Most recent generation job for this scene's clip.",
    )
    status = models.CharField(
        max_length=12,
        choices=[
            ("PENDING", "Pending"),
            ("RUNNING", "Running"),
            ("SUCCESS", "Success"),
            ("FAILED", "Failed"),
            ("CANCELLED", "Cancelled"),
        ],
        default="PENDING",
        help_text="Generation status of the scene's clip.",
    )

    class Transition(models.TextChoices):
        CUT = "cut", "Cut (hard cut)"
        FADE = "fade", "Fade"
        DISSOLVE = "dissolve", "Dissolve"
        WIPE_LEFT = "wipeleft", "Wipe left"
        WIPE_UP = "wipeup", "Wipe up"

    class Status(models.TextChoices):
        PENDING = "PENDING", "Pending"
        RUNNING = "RUNNING", "Running"
        SUCCESS = "SUCCESS", "Success"
        FAILED = "FAILED", "Failed"
        CANCELLED = "CANCELLED", "Cancelled"

    output_file = models.FileField(
        upload_to=_upload_to_films,
        blank=True,
        null=True,
        help_text="Generated scene clip.",
    )
    duration_seconds = models.FloatField(
        default=2.0,
        help_text="Scene duration in seconds.",
    )
    transition_in = models.CharField(
        max_length=32,
        blank=True,
        default=Transition.CUT,
        choices=Transition.choices,
        help_text="Transition before this scene.",
    )
    transition_out = models.CharField(
        max_length=32,
        blank=True,
        default=Transition.CUT,
        choices=Transition.choices,
        help_text="Transition after this scene.",
    )

    class Meta:
        ordering = ["order"]
        unique_together = [("project", "order")]

    def __str__(self) -> str:
        return f"Scene {self.id.hex[:8]} — {self.title or 'untitled'}"

    def get_status_display(self) -> str:
        return self.Status(self.status).label

    def status_label(self) -> str:
        return self.Status(self.status).label


class CharacterReference(models.Model):
    """Reusable character description and optional reference image."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    name = models.CharField(max_length=255, help_text="Character name.")
    description = models.TextField(help_text="Physical and stylistic description.")
    reference_image = models.ImageField(
        upload_to=_upload_to_films,
        blank=True,
        null=True,
        help_text="Reference image for the character, if supported by the model.",
    )
    notes = models.TextField(
        blank=True,
        help_text="Continuity notes, clothing, props, etc.",
    )

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return f"Character — {self.name}"

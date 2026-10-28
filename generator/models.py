"""
Generator app models — generation jobs and output videos.

CineForge AI — Local AI Video Generation Platform.

Job lifecycle (status field):
    PENDING   — queued, waiting for a worker slot
    RUNNING   — inference is in progress
    SUCCESS   — video generated and saved
    FAILED    — inference or post-processing failed
    CANCELLED — user or system cancelled the job
    DEFERRED  — queued but not startable yet (e.g. model not configured)

Timeouts and cancellations are handled in the Celery task and the service layer.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from django.conf import settings
from django.db import models


def _upload_to_outputs(instance: models.Model, filename: str) -> str:
    """Store generated videos under media/outputs/<uuid>/<filename>."""
    return str(
        settings.MEDIA_OUTPUTS_DIR / uuid.uuid4().hex / Path(filename).name
    )


def _upload_to_temp(instance: models.Model, filename: str) -> str:
    """Store temporary/intermediate files under media/temp/<uuid>/<filename>."""
    return str(settings.MEDIA_TEMP_DIR / uuid.uuid4().hex / Path(filename).name)


class GenerationJob(models.Model):
    """A single video generation request and its execution state."""

    class Status(models.TextChoices):
        PENDING = "PENDING", "Pending"
        RUNNING = "RUNNING", "Running"
        SUCCESS = "SUCCESS", "Success"
        FAILED = "FAILED", "Failed"
        CANCELLED = "CANCELLED", "Cancelled"
        DEFERRED = "DEFERRED", "Deferred"

    class Engine(models.TextChoices):
        COMFYUI = "COMFYUI", "ComfyUI"
        WAN = "WAN", "Wan (direct)"
        LTX = "LTX", "LTX-Video (direct)"
        TEST = "TEST", "Test adapter"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    # What the user asked for
    prompt = models.TextField(help_text="Natural-language generation prompt.")
    negative_prompt = models.TextField(
        blank=True, default="", help_text="Optional negative prompt."
    )

    # Selected model/engine
    engine = models.CharField(
        max_length=20,
        choices=Engine.choices,
        default=Engine.COMFYUI,
    )
    model_name = models.CharField(
        max_length=255,
        blank=True,
        help_text="User-visible model name (e.g. 'Wan 2.1 I2V').",
    )
    model_variant = models.CharField(
        max_length=64,
        blank=True,
        help_text="Optional model variant, e.g. 'FP16', 'Quantized'.",
    )

    # Generation parameters
    duration_seconds = models.FloatField(
        default=5.0,
        help_text="Requested video duration in seconds.",
    )
    width = models.PositiveIntegerField(
        default=832,
        help_text="Frame width in pixels.",
    )
    height = models.PositiveIntegerField(
        default=480,
        help_text="Frame height in pixels.",
    )
    fps = models.PositiveIntegerField(
        default=24,
        help_text="Frames per second.",
    )
    aspect_ratio = models.CharField(
        max_length=16,
        blank=True,
        help_text="Requested aspect ratio label (e.g. '16:9').",
    )
    seed = models.BigIntegerField(
        null=True,
        blank=True,
        help_text="Random seed for reproducibility.",
    )
    guidance_scale = models.FloatField(
        null=True,
        blank=True,
        help_text="Optional guidance/scale parameter if supported.",
    )
    steps = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Optional sampler steps if supported.",
    )

    # Job state
    status = models.CharField(
        max_length=12,
        choices=Status.choices,
        default=Status.PENDING,
    )
    task_id = models.CharField(
        max_length=255,
        blank=True,
        help_text="Celery task id once queued.",
    )
    progress = models.FloatField(
        default=0.0,
        help_text="Progress indicator 0.0-1.0 where reported.",
    )
    message = models.TextField(
        blank=True,
        help_text="Human-readable status / error message.",
    )

    # Output
    output_file = models.FileField(
        upload_to=_upload_to_outputs,
        blank=True,
        null=True,
        help_text="Generated video file when successful.",
    )
    error_type = models.CharField(
        max_length=64,
        blank=True,
        help_text="Machine-readable error category.",
    )
    log_snippet = models.TextField(
        blank=True,
        help_text="Short excerpt from the run log.",
    )

    class Meta:
        ordering = ["-created"]
        indexes = [
            models.Index(fields=["status"]),
            models.Index(fields=["-created"]),
        ]

    # Fields that live on the job but are not part of the core generation spec.
    # They are used for image-to-video and film-scene workflows.
    reference_image = models.ImageField(
        upload_to=_upload_to_temp,
        blank=True,
        null=True,
        help_text="Reference image for image-to-video jobs.",
    )
    temporary_image_path = models.CharField(
        max_length=512,
        blank=True,
        help_text="Temp path for uploaded image that has not been cleaned up yet.",
    )

    def __str__(self) -> str:
        return f"Job {self.id.hex[:8]} — {self.get_status_display()}"

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            self.Status.SUCCESS,
            self.Status.FAILED,
            self.Status.CANCELLED,
        }

    @property
    def output_path(self) -> Path | None:
        if self.output_file:
            return Path(self.output_file.path)
        return None

    def can_retry(self) -> bool:
        return self.status in {self.Status.FAILED, self.Status.CANCELLED}

    def can_be_cancelled(self) -> bool:
        return self.status in {self.Status.PENDING, self.Status.RUNNING}

    def can_generate_without_files(self) -> bool:
        """Return True when the job does not require an uploaded reference image."""
        # Jobs marked explicitly as image-to-video must carry a reference.
        return True

    def create_retry(self) -> GenerationJob:
        """Create a new failed/cancelled job with the same generation parameters."""
        new = GenerationJob.objects.create(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            engine=self.engine,
            model_name=self.model_name,
            model_variant=self.model_variant,
            duration_seconds=self.duration_seconds,
            width=self.width,
            height=self.height,
            fps=self.fps,
            aspect_ratio=self.aspect_ratio,
            seed=self.seed,
            guidance_scale=self.guidance_scale,
            steps=self.steps,
            status=GenerationJob.Status.PENDING,
            message="Retry of a previous job.",
        )
        return new


class GeneratedVideo(models.Model):
    """A generated video asset, separate from the job that produced it.

    This keeps a clean library view even when jobs are deleted.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    job = models.OneToOneField(
        GenerationJob,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="generated_video",
    )
    title = models.CharField(
        max_length=255,
        blank=True,
        help_text="User-editable title.",
    )
    prompt_summary = models.TextField(
        blank=True,
        help_text="Short summary of the prompt for library display.",
    )
    file = models.FileField(
        upload_to=_upload_to_outputs,
        help_text="Video file.",
    )
    thumbnail = models.ImageField(
        upload_to=_upload_to_temp,
        blank=True,
        null=True,
        help_text="Auto-generated thumbnail frame.",
    )

    # Metadata detected/cached from the file
    duration_seconds = models.FloatField(
        null=True,
        blank=True,
        help_text="Duration in seconds.",
    )
    width = models.PositiveIntegerField(
        null=True,
        blank=True,
    )
    height = models.PositiveIntegerField(
        null=True,
        blank=True,
    )
    fps = models.PositiveIntegerField(
        null=True,
        blank=True,
    )
    codec = models.CharField(
        max_length=64,
        blank=True,
        help_text="Container or video codec name.",
    )
    size_bytes = models.BigIntegerField(
        null=True,
        blank=True,
        help_text="File size in bytes.",
    )

    class Meta:
        ordering = ["-created"]

    def __str__(self) -> str:
        return f"Video {self.id.hex[:8]} — {self.title or 'untitled'}"

    @property
    def size_mb(self) -> float:
        if self.size_bytes:
            return round(self.size_bytes / (1024 * 1024), 2)
        return 0.0

    @property
    def duration_display(self) -> str:
        if self.duration_seconds is None:
            return "—"
        total = round(self.duration_seconds)
        m, s = divmod(total, 60)
        return f"{m}:{s:02d}"

    def natural_size(self) -> str:
        if not self.size_bytes:
            return "—"
        size = self.size_bytes
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def detect_metadata(self) -> None:
        """Populate cached metadata from the stored file using ffprobe.

        No-op when ffmpeg/ffprobe is not available or the file is missing.
        """
        from generator.services import probe_video_file

        if not self.file or not self.file.path:
            return
        path = Path(self.file.path)
        if not path.exists():
            return
        meta = probe_video_file(path)
        if meta:
            self.duration_seconds = meta.get("duration")
            self.width = meta.get("width")
            self.height = meta.get("height")
            self.fps = meta.get("fps")
            self.codec = meta.get("codec")
            self.size_bytes = meta.get("size")
            self.save(update_fields=[
                "duration_seconds",
                "width",
                "height",
                "fps",
                "codec",
                "size_bytes",
                "updated",
            ])

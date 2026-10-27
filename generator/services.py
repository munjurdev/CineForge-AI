"""
Generator service layer.

CineForge AI — Local AI Video Generation Platform.

This module is the main integration point between the Django layer and the
inference adapters. It owns:
- job validation and normalisation
- queueing via Celery (with graceful degradation to run-in-process when Celery
  is configured with ``CELERY_TASK_ALWAYS_EAGER``)
- cancellation bookkeeping
- video file probing via ffprobe
"""
from __future__ import annotations

import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db import close_old_connections

from .models import GenerationJob

logger = logging.getLogger("cineforge.generator")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

# Common model presets keyed by engine name. These are suggestions, not hard
# limits — real limits come from the installed adapter and hardware.
PRESETS: dict[str, dict[str, Any]] = {
    "COMFYUI": {
        "width": 832,
        "height": 480,
        "fps": 24,
        "duration_seconds": 2.0,
        "aspect_ratio": "16:9",
        "guidance_scale": 7.5,
        "steps": 25,
    },
    "WAN": {
        "width": 832,
        "height": 480,
        "fps": 16,
        "duration_seconds": 5.0,
        "aspect_ratio": "16:9",
        "guidance_scale": 5.0,
        "steps": 50,
    },
    "LTX": {
        "width": 512,
        "height": 512,
        "fps": 24,
        "duration_seconds": 2.0,
        "aspect_ratio": "1:1",
        "guidance_scale": 3.5,
        "steps": 30,
    },
    "TEST": {
        "width": 64,
        "height": 64,
        "fps": 12,
        "duration_seconds": 0.5,
        "aspect_ratio": "1:1",
        "guidance_scale": 5.0,
        "steps": 1,
    },
}


def validate_and_normalize_job(
    job: GenerationJob,
    *,
    skip_files: bool = False,
) -> tuple[bool, str | None, list[str]]:
    """Validate the job and return (ok, message, warnings).

    Normalises missing values from presets when possible. The job is NOT saved
    by this function; callers should call job.save() after review.
    """
    warnings: list[str] = []

    preset = PRESETS.get(job.engine)

    if preset:
        # Fill missing values from preset
        if not job.model_name:
            job.model_name = _suggest_model_name(job.engine)
        if job.duration_seconds is None:
            job.duration_seconds = preset["duration_seconds"]
        if job.width is None:
            job.width = preset["width"]
        if job.height is None:
            job.height = preset["height"]
        if job.fps is None:
            job.fps = preset["fps"]
        if not job.aspect_ratio:
            job.aspect_ratio = preset["aspect_ratio"]
        if job.guidance_scale is None:
            job.guidance_scale = preset.get("guidance_scale")
        if job.steps is None:
            job.steps = preset.get("steps")

    # Basic sanity
    if not job.prompt or not job.prompt.strip():
        return False, "A prompt is required.", []

    if job.width is not None:
        if job.width < 64:
            return False, "Width must be at least 64 pixels.", []
        if job.width > 2048:
            return False, "Width must be at most 2048 pixels for local generation.", []
        if job.width % 8:
            warnings.append(
                "Width should ideally be a multiple of 8 for most video models."
            )

    if job.height is not None:
        if job.height < 64:
            return False, "Height must be at least 64 pixels.", []
        if job.height > 2048:
            return False, "Height must be at most 2048 pixels for local generation.", []
        if job.height % 8:
            warnings.append(
                "Height should ideally be a multiple of 8 for most video models."
            )

    if job.duration_seconds is not None:
        if job.duration_seconds <= 0:
            return False, "Duration must be greater than zero.", []
        if job.duration_seconds > 30:
            warnings.append(
                "Longer durations increase VRAM and time requirements."
            )

    if job.fps is not None:
        if job.fps < 1 or job.fps > 60:
            return False, "FPS must be between 1 and 60.", []

    if job.guidance_scale is not None:
        if job.guidance_scale < 0 or job.guidance_scale > 20:
            return False, "Guidance scale must be between 0 and 20.", []

    if job.steps is not None:
        if job.steps < 1 or job.steps > 500:
            return False, "Steps must be between 1 and 500.", []

    # Reference-image requirement is enforced by forms and the image-to-video
    # flow; this service layer does not reject jobs that lack an image here.

    return True, None, warnings


def get_engine_status_summary() -> dict[str, Any]:
    """Summarise the configured inference engine for the dashboard."""
    from .adapters import registry

    configured = getattr(settings, "INFERENCE_DEFAULT_ADAPTER", "comfyui").lower()
    adapters = registry.names()
    labels = {
        "comfyui": "ComfyUI",
        "wan": "Wan",
        "ltx": "LTX-Video",
        "test": "Test adapter",
    }
    return {
        "configured": configured,
        "label": labels.get(configured, configured.title()),
        "adapters": adapters,
        "ready": configured in adapters,
    }


def _suggest_model_name(engine: str) -> str:
    """Return a human-readable default model name for the engine."""
    return {
        "COMFYUI": "ComfyUI workflow (select model in ComfyUI)",
        "WAN": "Wan 2.1 (configure locally)",
        "LTX": "LTX-Video (configure locally)",
        "TEST": "Test adapter (CPU-only demo)",
    }.get(engine, engine)


# ---------------------------------------------------------------------------
# Forms bridge
# ---------------------------------------------------------------------------

def validate_and_normalize_job_from_form(
    form: Any,
    user: Any | None = None,
) -> GenerationJob:
    """Build, validate, and return a GenerationJob from a validated form."""
    job = GenerationJob()
    job.prompt = (form.cleaned_data.get("prompt") or "").strip()
    job.negative_prompt = (form.cleaned_data.get("negative_prompt") or "").strip()
    job.engine = form.cleaned_data.get("engine") or GenerationJob.Engine.COMFYUI
    job.model_name = (form.cleaned_data.get("model_name") or "").strip()
    job.duration_seconds = form.cleaned_data.get("duration_seconds", 2.0)
    job.width = form.cleaned_data.get("width", 832)
    job.height = form.cleaned_data.get("height", 480)
    job.fps = form.cleaned_data.get("fps", 24)
    job.aspect_ratio = (form.cleaned_data.get("aspect_ratio") or "").strip()
    job.seed = form.cleaned_data.get("seed")
    job.guidance_scale = form.cleaned_data.get("guidance_scale")
    job.steps = form.cleaned_data.get("steps")
    job.model_variant = (form.cleaned_data.get("model_variant") or "").strip()

    # Carry image across when provided
    if hasattr(form, "cleaned_data") and "reference_image" in form.cleaned_data:
        image = form.cleaned_data.get("reference_image")
        if image:
            job.reference_image = image

    # Handle temporary storage for uploaded image via form hook
    if hasattr(form, "save_temporary_image"):
        saved = form.save_temporary_image(job)
        if saved:
            job.temporary_image_path = saved

    ok, msg, warnings = validate_and_normalize_job(job)
    if not ok:
        raise ValueError(msg)

    # Attach user-facing warnings back to the job for the detail view
    job._warnings = warnings  # noqa: SLF001
    return job


# ---------------------------------------------------------------------------
# Queueing / cancellation
# ---------------------------------------------------------------------------

def _run_eager_task_in_background(job_id: str, task_id: str) -> None:
    """Run eager Celery tasks off the request thread for the local web UI."""
    from .tasks import run_generation_task

    close_old_connections()
    try:
        run_generation_task.apply(args=(job_id,), task_id=task_id)
    finally:
        close_old_connections()


def queue_generation_job(
    job: GenerationJob,
    *,
    background: bool = False,
) -> tuple[bool, str | None]:
    """Queue the job for generation and return (queued, message)."""
    from .tasks import run_generation_task

    # UUID primary keys are assigned on instantiation, so a form-built job can
    # have a pk without ever being written to the database.
    if job._state.adding or job.pk is None:
        job.save()

    # Guard: don't queue terminal jobs
    if job.is_terminal:
        return False, "Job is already finished."

    if job.status == GenerationJob.Status.RUNNING:
        return False, "Job is already running."

    try:
        if background and getattr(settings, "CELERY_TASK_ALWAYS_EAGER", False):
            task_id = str(uuid.uuid4())
            job.task_id = task_id
            job.status = GenerationJob.Status.RUNNING
            job.progress = 0.0
            job.message = "Starting generation in the background…"
            job.save(
                update_fields=[
                    "task_id",
                    "status",
                    "progress",
                    "message",
                    "updated",
                ]
            )
            worker = threading.Thread(
                target=_run_eager_task_in_background,
                args=(str(job.id), task_id),
                name=f"generation-{job.id.hex[:8]}",
                daemon=True,
            )
            worker.start()
            logger.info("Started background generation job %s", job.id)
            return True, "Generation started. This page will show live progress."

        result = run_generation_task.delay(job_id=str(job.id))
        job.task_id = result.id

        # In eager mode the task has already run and written a terminal state;
        # refreshing avoids clobbering it with a stale RUNNING status.
        job.refresh_from_db(
            fields=["status", "progress", "message", "output_file", "error_type"]
        )
        update_fields = ["task_id", "updated"]
        if job.status == GenerationJob.Status.PENDING:
            job.status = GenerationJob.Status.RUNNING
            job.progress = 0.0
            job.message = "Generation started."
            update_fields += ["status", "progress", "message"]
        job.save(update_fields=update_fields)
        logger.info("Queued generation job %s with task %s", job.id, result.id)
        return True, f"Generation queued (task {result.id})."
    except Exception as exc:  # pragma: no cover - network/redis
        job.status = GenerationJob.Status.FAILED
        job.message = f"Could not queue job: {exc}"
        job.error_type = "QUEUE_ERROR"
        job.save(update_fields=["status", "message", "error_type", "updated"])
        logger.error("Failed to queue job %s: %s", job.id, exc, exc_info=True)
        return False, f"Could not queue job: {exc}"


def cancel_generation_job(job: GenerationJob) -> tuple[bool, str | None]:
    """Mark a job as cancelled and attempt to revoke the Celery task."""
    from .tasks import revoke_task

    if not job.can_be_cancelled():
        return False, "Job cannot be cancelled."

    job.status = GenerationJob.Status.CANCELLED
    job.message = "Job cancelled by user."
    job.save(update_fields=["status", "message", "updated"])
    sync_film_scenes_for_job(job)

    if job.task_id:
        try:
            revoke_task(job.task_id)
            logger.info("Revoked task %s for job %s", job.task_id, job.id)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not revoke task %s: %s", job.task_id, exc)

    return True, "Job cancelled."


# ---------------------------------------------------------------------------
# Probing / file helpers
# ---------------------------------------------------------------------------

def probe_video_file(path: Path) -> dict[str, Any] | None:
    """Return metadata for a video file via ffprobe, or None on failure."""
    if not path.exists():
        return None

    bin_name = getattr(settings, "FFMPEG_BIN", "ffmpeg")
    ffprobe = bin_name.replace("ffmpeg", "ffprobe")
    if not shutil.which(ffprobe):
        # fallback: try the configured binary directly with -version for probe
        ffprobe = bin_name

    if not shutil.which(ffprobe):
        logger.debug("ffprobe not found; skipping probe for %s", path)
        return None

    import subprocess

    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            logger.debug("ffprobe failed for %s: %s", path, proc.stderr)
            return None
        import json

        data = json.loads(proc.stdout)
    except Exception as exc:  # pragma: no cover - subprocess/json
        logger.debug("Could not probe %s: %s", path, exc)
        return None

    fmt = data.get("format", {})
    streams = data.get("streams", [])
    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"), streams[0] if streams else {}
    )

    def _float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return {
        "duration": _float(fmt.get("duration")),
        "width": _int(video_stream.get("width")),
        "height": _int(video_stream.get("height")),
        "fps": _fps_from_stream(video_stream),
        "codec": video_stream.get("codec_name", ""),
        "size": _int(fmt.get("size")),
        "format_name": fmt.get("format_name", ""),
    }


def _fps_from_stream(stream: dict[str, Any]) -> int | None:
    """Derive a sensible integer FPS from ffprobe stream data."""
    # r_frame_rate is like "24/1"
    rate = stream.get("r_frame_rate", "")
    if isinstance(rate, str) and "/" in rate:
        num, den = rate.split("/")
        try:
            num = int(num)
            den = int(den)
            if den and num:
                return round(num / den)
        except ValueError:
            pass
    # Fallback to coded_drop or avg_frame_rate
    avg = stream.get("avg_frame_rate", "")
    if isinstance(avg, str) and "/" in avg:
        num, den = avg.split("/")
        try:
            return round(int(num) / int(den))
        except (ValueError, ZeroDivisionError):
            pass
    return None


def delete_job_files(job: GenerationJob) -> None:
    """Delete stored files for a job (output and any temp reference)."""
    if job.output_file and job.output_file.path:
        _safe_delete_file(job.output_file.path)
        job.output_file.delete(save=False)

    if job.temporary_image_path:
        _safe_delete_file(job.temporary_image_path)

    job.temporary_image_path = ""
    job.save(update_fields=["temporary_image_path", "updated"])


def _safe_delete_file(path: str) -> None:
    """Delete a file only if it lives under MEDIA_ROOT."""
    try:
        p = Path(path).resolve()
        media_root = settings.MEDIA_ROOT.resolve()
        if not str(p).startswith(str(media_root)):
            logger.warning(
                "Refusing to delete file outside MEDIA_ROOT: %s", path
            )
            return
        if p.exists():
            p.unlink()
            logger.info("Deleted file: %s", p)
    except OSError as exc:
        logger.warning("Could not delete file %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Job lifecycle helpers (used by views)
# ---------------------------------------------------------------------------

def error_is_user_actionable(job: GenerationJob) -> bool:
    """Heuristic: is the failure something the user can fix by changing inputs?"""
    t = (job.error_type or "").upper()
    m = (job.message or "").lower()
    user_errors = {
        "INVALID_PROMPT",
        "INVALID_SETTINGS",
        "MISSING_IMAGE",
        "MISSING_MODEL",
        "MODEL_NOT_CONFIGURED",
        "RESOLUTION_OUT_OF_RANGE",
        "WARMUP_FAILED",
    }
    if t in user_errors:
        return True
    return any(
        keyword in m
        for keyword in (
            "missing model",
            "not configured",
            "invalid",
            "unsupported",
            "image",
            "file",
        )
    )


# ---------------------------------------------------------------------------
# Module-level predicate registration for adapters
# ---------------------------------------------------------------------------

# These are called by the adapter layer at execution time and mutate the job.
def job_set_progress(job: GenerationJob, value: float, message: str | None = None) -> None:
    """Update progress without racing with other writes."""
    clamped = max(0.0, min(1.0, float(value)))
    job.progress = round(clamped, 3)
    if message:
        job.message = message
    job.save(update_fields=["progress", "message", "updated"])


def job_fail(job: GenerationJob, message: str, error_type: str = "INTERNAL_ERROR") -> None:
    job.status = GenerationJob.Status.FAILED
    job.message = message
    job.error_type = error_type
    job.save(update_fields=["status", "message", "error_type", "updated"])
    sync_film_scenes_for_job(job)


def job_succeed(job: GenerationJob, output_path: Path) -> None:
    job.status = GenerationJob.Status.SUCCESS
    job.message = "Generation completed."
    job.progress = 1.0

    # Attach output file
    if job.output_file:
        job.output_file.delete(save=False)
    # Move/copy the output into the proper media location
    dest = _copy_or_move_to_media_output(job, output_path)
    if dest:
        job.output_file = dest
    job.save(update_fields=["status", "message", "progress", "output_file", "updated"])
    sync_film_scenes_for_job(job)


def sync_film_scenes_for_job(job: GenerationJob, video=None) -> None:
    """Update any film scenes linked to this job, without breaking generation."""
    try:
        from films.services import sync_scene_from_job
    except Exception:  # pragma: no cover - import guard
        return
    try:
        sync_scene_from_job(job, video=video)
    except Exception:  # pragma: no cover - bookkeeping must not fail a job
        logger.debug(
            "Could not sync film scenes for job %s", job.id, exc_info=True
        )


def _copy_or_move_to_media_output(job: GenerationJob, src: Path) -> str | None:
    """Copy or move the generated file into media/outputs and return the storage path."""
    if not src.exists():
        logger.warning("Output file missing for job %s: %s", job.id, src)
        return None

    # Create a deterministic subfolder based on job id
    folder = settings.MEDIA_OUTPUTS_DIR / job.id.hex
    folder.mkdir(parents=True, exist_ok=True)

    dest_path = folder / src.name
    try:
        shutil.move(str(src), str(dest_path))
        logger.info("Moved output %s -> %s", src, dest_path)
    except OSError:
        shutil.copy2(str(src), str(dest_path))
        logger.info("Copied output %s -> %s", src, dest_path)

    return str(dest_path)

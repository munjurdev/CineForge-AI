"""
Celery tasks for video generation and status reporting.

CineForge AI — Local AI Video Generation Platform.

Jobs are dispatched via ``queue_generation_job`` in services.py. This module
contains the worker-side task that actually invokes the adapter.
"""
from __future__ import annotations

import logging
from pathlib import Path

from celery import shared_task
from django.conf import settings

from .adapters import get_adapter
from .audio import add_voiceover_to_video
from .models import GenerationJob
from .services import (
    job_fail,
    job_set_progress,
    job_succeed,
    sync_film_scenes_for_job,
)

logger = logging.getLogger("cineforge.generator.tasks")


def revoke_task(task_id: str) -> None:
    """Revoke a Celery task by id if the app supports it."""
    from celery import current_app

    app = current_app
    if app is None:
        return
    try:
        app.control.revoke(task_id, terminate=True, signal="SIGTERM")
    except Exception as exc:  # pragma: no cover - control api
        logger.debug("Could not revoke task %s: %s", task_id, exc)


@shared_task(name="cineforge.generator.run_generation_task")
def run_generation_task(job_id: str) -> dict[str, str]:
    """Run a generation job via the configured adapter.

    This is the main worker entry point. It is tolerant of missing models and
    reports the real outcome back to the job record.
    """
    try:
        job = GenerationJob.objects.get(pk=job_id)
    except GenerationJob.DoesNotExist:
        logger.error("GenerationTask received unknown job_id=%s", job_id)
        return {"status": "error", "message": "Unknown job"}

    adapter_name = job.engine
    try:
        adapter = get_adapter(adapter_name)
    except Exception as exc:  # pragma: no cover - adapter loading
        job_fail(job, f"Adapter error: {exc}", error_type="ADAPTER_ERROR")
        return {"status": "error", "message": f"Adapter error: {exc}"}

    logger.info(
        "Starting generation %s with adapter=%s model=%s",
        job.id,
        adapter_name,
        job.model_name,
    )
    job_set_progress(job, 0.0, "Generation started.")

    try:
        # If the adapter needs to validate the job, do that first
        try:
            ok, msg, warnings = adapter.validate(job)
        except NotImplementedError:
            ok, msg, warnings = True, None, []

        if not ok:
            job_fail(job, msg or "Adapter validation failed.", error_type="ADAPTER_REJECTED")
            return {"status": "error", "message": msg}

        if warnings:
            job.message = msg if msg else "Adapter warnings."
            job.save(update_fields=["message"])

        # Run inference
        result = adapter.generate(job)

        if result is None:
            job_fail(job, "Adapter returned no result.", error_type="ADAPTER_NO_OUTPUT")
            return {"status": "error", "message": "Adapter returned no result."}

        output = Path(result)

        if not output.exists():
            job_fail(job, f"Output file not found: {output}", error_type="MISSING_OUTPUT")
            return {"status": "error", "message": f"Output file not found: {output}"}

        # The video workflows render silent frames only. Add the spoken
        # narration here so every generated clip comes out with a voice track.
        if getattr(settings, "VOICEOVER_ENABLED", True):
            job_set_progress(job, 0.96, "Adding AI voiceover…")
            output = add_voiceover_to_video(job, output)

        job_succeed(job, output)

        # Auto-create library entry
        from .models import GeneratedVideo

        try:
            video = GeneratedVideo.objects.create(
                job=job,
                file=job.output_file,
                title=(job.prompt or "").split("\n")[0][:120] or "untitled",
                prompt_summary=(job.prompt or "")[:250],
            )
            video.detect_metadata()
            sync_film_scenes_for_job(job, video)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not create library entry for job %s: %s", job.id, exc)

        logger.info("Generation %s succeeded: %s", job.id, output)
        return {"status": "success", "output": str(output)}

    except Exception as exc:  # noqa: BLE001
        logger.error("Generation %s failed: %s", job.id, exc, exc_info=True)
        job_fail(
            job,
            f"Generation failed: {exc}",
            error_type="INTERNAL_ERROR",
        )
        return {"status": "error", "message": str(exc)}


def report_job_progress(job_id: str, progress: float, message: str | None = None) -> None:
    """Celery-side helper to push progress updates back to the job."""
    try:
        job = GenerationJob.objects.get(pk=job_id)
    except GenerationJob.DoesNotExist:
        return
    job_set_progress(job, progress, message)

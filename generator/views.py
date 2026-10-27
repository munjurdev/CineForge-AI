"""
Generator app views — submission, status, cancel, retry, delete.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

import logging

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .forms import GenerationJobForm, ImageToVideoForm
from .models import GenerationJob
from .prompt_assistant import METHOD as PROMPT_ASSISTANT_METHOD
from .prompt_assistant import build_cinematic_prompt
from .services import (
    cancel_generation_job,
    delete_job_files,
    error_is_user_actionable,
    queue_generation_job,
    validate_and_normalize_job_from_form,
)

logger = logging.getLogger("cineforge.generator")


def generate_video(request):
    """Text-to-video submission page."""
    if request.method == "POST":
        form = GenerationJobForm(request.POST)
        if form.is_valid():
            try:
                job = validate_and_normalize_job_from_form(form, request.user)
            except ValueError as exc:
                messages.error(request, str(exc))
                return render(request, "generator/generate_video.html", {
                    "page": "generate_video",
                    "form": form,
                })

            for warning in getattr(job, "_warnings", []):
                messages.warning(request, warning)

            queued, reason = queue_generation_job(job, background=True)
            if queued:
                messages.success(request, reason or "Generation queued.")
                return redirect("generator:job_detail", pk=job.pk)
            else:
                messages.warning(request, reason or "Generation could not be queued.")
                return render(request, "generator/generate_video.html", {
                    "page": "generate_video",
                    "form": form,
                    "job": job,
                })
    else:
        form = GenerationJobForm()

    return render(request, "generator/generate_video.html", {
        "page": "generate_video",
        "form": form,
    })


def generate_image_to_video(request):
    """Image-to-video submission page."""
    if request.method == "POST":
        form = ImageToVideoForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                job = validate_and_normalize_job_from_form(form, request.user)
            except ValueError as exc:
                messages.error(request, str(exc))
                return render(request, "generator/generate_image_to_video.html", {
                    "page": "image_to_video",
                    "form": form,
                })

            for warning in getattr(job, "_warnings", []):
                messages.warning(request, warning)

            queued, reason = queue_generation_job(job, background=True)
            if queued:
                messages.success(request, reason or "Image-to-video queued.")
                return redirect("generator:job_detail", pk=job.pk)
            else:
                messages.warning(request, reason or "Generation could not be queued.")
                return render(request, "generator/generate_image_to_video.html", {
                    "page": "image_to_video",
                    "form": form,
                    "job": job,
                })
    else:
        form = ImageToVideoForm()

    return render(request, "generator/generate_image_to_video.html", {
        "page": "image_to_video",
        "form": form,
    })


def job_detail(request, pk):
    """Show a job's current status and available actions."""
    job = get_object_or_404(GenerationJob, pk=pk)
    progress_known = (
        job.status == GenerationJob.Status.SUCCESS
        or job.message.startswith("Sampling step ")
    )
    return render(request, "generator/job_detail.html", {
        "page": "job_detail",
        "job": job,
        "is_error_actionable": error_is_user_actionable(job),
        "progress_percent": round((job.progress or 0.0) * 100),
        "progress_known": progress_known,
    })


def job_status(request, pk):
    """Return the latest generation state for the live progress page."""
    job = get_object_or_404(GenerationJob, pk=pk)
    progress_known = (
        job.status == GenerationJob.Status.SUCCESS
        or job.message.startswith("Sampling step ")
    )
    return JsonResponse({
        "status": job.status,
        "status_display": job.get_status_display(),
        "message": job.message,
        "progress_percent": round((job.progress or 0.0) * 100),
        "progress_known": progress_known,
    })


def cancel_job(request, pk):
    """Cancel a running or pending job."""
    job = get_object_or_404(GenerationJob, pk=pk)
    if not job.can_be_cancelled():
        messages.warning(request, "This job cannot be cancelled in its current state.")
        return redirect("generator:job_detail", pk=job.pk)

    cancelled, reason = cancel_generation_job(job)
    if cancelled:
        messages.success(request, reason or "Job cancelled.")
    else:
        messages.warning(request, reason or "Could not cancel job.")

    return redirect("generator:job_detail", pk=job.pk)


def retry_job(request, pk):
    """Re-submit a failed or cancelled job with the same parameters."""
    job = get_object_or_404(GenerationJob, pk=pk)
    if not job.can_retry():
        messages.warning(request, "Only failed or cancelled jobs can be retried.")
        return redirect("generator:job_detail", pk=job.pk)

    new_job = job.create_retry()
    _relink_film_scenes(job, new_job)
    queued, reason = queue_generation_job(new_job)
    if queued:
        messages.success(request, reason or "Job queued for retry.")
    else:
        messages.warning(request, reason or "Retry job created but could not be queued.")
    return redirect("generator:job_detail", pk=new_job.pk)


def delete_job(request, pk):
    """Delete a job and its associated output files."""
    job = get_object_or_404(GenerationJob, pk=pk)
    delete_job_files(job)
    job.delete()
    messages.success(request, "Job and its files were deleted.")
    return redirect("generator:job_list")


def _relink_film_scenes(old_job: GenerationJob, new_job: GenerationJob) -> None:
    """Keep film scenes attached to a scene job when it is retried."""
    try:
        from films.models import FilmScene
    except Exception:  # pragma: no cover - import guard
        return
    FilmScene.objects.filter(job=old_job).update(
        job=new_job, status=FilmScene.Status.PENDING
    )


@require_POST
def enhance_prompt(request):
    """Expand a short idea with the offline, rule-based prompt assistant."""
    try:
        prompt = build_cinematic_prompt(request.POST.get("prompt", ""))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    return JsonResponse({"prompt": prompt, "method": PROMPT_ASSISTANT_METHOD})


def job_list(request):
    """List recent generation jobs with their current status."""
    jobs = GenerationJob.objects.select_related(
        "generated_video",
    ).order_by("-updated")[:50]
    return render(request, "generator/job_list.html", {
        "page": "job_list",
        "jobs": jobs,
    })

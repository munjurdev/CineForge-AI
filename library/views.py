"""
Library app views — video gallery, detail, download, and delete.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.http import (
    FileResponse,
    HttpResponseBadRequest,
    HttpResponseNotFound,
)
from django.shortcuts import get_object_or_404, redirect, render

from generator.models import GeneratedVideo

logger = logging.getLogger("cineforge.library")


def _disk_usage_summary() -> dict[str, float]:
    """Return media volume usage in GB for the library header."""
    try:
        usage = shutil.disk_usage(settings.MEDIA_ROOT)
    except OSError:
        return {}
    gb = 1024 ** 3
    return {
        "used_gb": round(usage.used / gb, 1),
        "total_gb": round(usage.total / gb, 1),
        "free_gb": round(usage.free / gb, 1),
    }


def video_list(request):
    """Render the video gallery."""
    videos = GeneratedVideo.objects.select_related("job").order_by("-created")
    return render(request, "library/video_list.html", {
        "page": "video_list",
        "videos": videos,
        "disk_usage": _disk_usage_summary(),
    })


def video_detail(request, pk):
    """Show a single video with its metadata and actions."""
    video = get_object_or_404(GeneratedVideo, pk=pk)
    return render(request, "library/video_detail.html", {
        "page": "video_detail",
        "video": video,
    })


def video_download(request, pk):
    """Serve the stored video file for download."""
    video = get_object_or_404(GeneratedVideo, pk=pk)
    if not video.file or not video.file.path:
        return HttpResponseNotFound("Video file not found.")

    path = Path(video.file.path)
    if not path.exists():
        return HttpResponseNotFound("Video file not found on disk.")

    try:
        response = FileResponse(
            path.open("rb"),
            as_attachment=True,
            filename=path.name,
        )
        response["Content-Type"] = "video/mp4"
        return response
    except OSError as exc:
        logger.warning("Could not stream %s: %s", path, exc)
        return HttpResponseBadRequest("Could not serve video file.")


def video_delete(request, pk):
    """Delete a video and its associated files (local single-user app)."""
    video = get_object_or_404(GeneratedVideo, pk=pk)
    for field in (video.file, video.thumbnail):
        if not field:
            continue
        try:
            field.delete(save=False)
        except OSError as exc:
            logger.warning("Could not delete media file %s: %s", field.name, exc)
    video.delete()
    messages.success(request, "Video was deleted.")
    return redirect("library:video_list")

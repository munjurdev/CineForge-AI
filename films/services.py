"""
Film assembly service — joins generated scene clips with FFmpeg.

CineForge AI — Local AI Video Generation Platform.

This module is intentionally thin: it validates inputs, builds an FFmpeg
command, runs it, and records the outcome on the project. It does not promise
perfect scene continuity or character consistency; those depend on the
underlying model and workflow, not on this assembler.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from django.conf import settings

from .models import FilmProject, FilmScene

logger = logging.getLogger("cineforge.films")

# FFmpeg xfade transitions we expose in the UI. Anything else is a hard cut.
KNOWN_TRANSITIONS = {"fade", "dissolve", "wipeleft", "wipeup"}


def _normalise_transition(value: str | None) -> str:
    """Return a supported xfade transition name, or 'cut' for unknown values."""
    value = (value or "").strip().lower()
    return value if value in KNOWN_TRANSITIONS else "cut"


def _even(value: Any) -> int:
    """Coerce a dimension to a positive even integer for yuv420p encoding."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 832
    number = max(2, number)
    return number if number % 2 == 0 else number - 1


def build_xfade_filtergraph(
    durations: list[float],
    transitions: list[str],
    *,
    width: int,
    height: int,
    fps: int,
    transition_seconds: float,
) -> str | None:
    """Build an FFmpeg filter_complex graph with crossfade transitions.

    Returns ``None`` when every transition is a hard cut or when clip
    durations are unknown, so the caller can keep the fast concat path.
    The final video label is ``outv``.
    """
    if len(durations) < 2:
        return None
    if any(duration <= 0 for duration in durations):
        return None

    cut_chain = [_normalise_transition(item) for item in transitions]
    cut_chain = cut_chain[: len(durations) - 1]
    while len(cut_chain) < len(durations) - 1:
        cut_chain.append("cut")

    if all(item == "cut" for item in cut_chain):
        return None

    parts: list[str] = []
    for index in range(len(durations)):
        parts.append(
            f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
            f"format=yuv420p,setpts=PTS-STARTPTS[v{index}]"
        )

    current = "v0"
    elapsed = durations[0]
    for index, transition in enumerate(cut_chain, start=1):
        duration = min(
            float(transition_seconds),
            durations[index - 1] / 2,
            durations[index] / 2,
        )
        if transition == "cut" or duration <= 0:
            label = f"c{index}"
            parts.append(f"[{current}][v{index}]concat=n=2:v=1:a=0[{label}]")
            elapsed += durations[index]
        else:
            offset = max(0.0, elapsed - duration)
            label = f"x{index}"
            parts.append(
                f"[{current}][v{index}]xfade=transition={transition}:"
                f"duration={duration:.3f}:offset={offset:.3f}[{label}]"
            )
            elapsed = elapsed + durations[index] - duration
        current = label

    parts.append(f"[{current}]null[outv]")
    return ";".join(parts)


def assemble_film(project: FilmProject) -> dict[str, Any]:
    """Run FFmpeg to assemble the project's scene clips into one movie.

    Returns a dict with at least 'success' (bool) and 'message' (str).
    """
    from generator.services import probe_video_file

    if not project.can_assemble():
        msg = "Film needs at least two successfully generated scenes before it can be assembled."
        project.assembly_error = msg
        project.save(update_fields=["assembly_error", "updated"])
        return {"success": False, "message": msg}

    scenes = list(
        project.scenes.filter(
            status="SUCCESS",
            output_file__isnull=False,
        ).order_by("order")
    )

    if len(scenes) < 2:
        msg = "Not enough successful scenes were found for assembly."
        project.assembly_error = msg
        project.save(update_fields=["assembly_error", "updated"])
        return {"success": False, "message": msg}

    ffmpeg_bin = getattr(settings, "FFMPEG_BIN", "ffmpeg")
    if not shutil.which(ffmpeg_bin):
        msg = "FFmpeg is not available on this system."
        project.assembly_error = msg
        project.save(update_fields=["assembly_error", "updated"])
        return {"success": False, "message": msg}

    input_files: list[Path] = []
    metadata: list[dict[str, Any]] = []
    for scene in scenes:
        if not scene.output_file or not scene.output_file.path:
            continue
        p = Path(scene.output_file.path)
        if not p.exists():
            continue
        input_files.append(p)
        metadata.append(probe_video_file(p) or {})

    if len(input_files) < 2:
        msg = "No valid scene clips were found on disk for assembly."
        project.assembly_error = msg
        project.save(update_fields=["assembly_error", "updated"])
        return {"success": False, "message": msg}

    transitions = [
        scene.transition_out for scene in scenes[: len(input_files) - 1]
    ]
    durations = [float(meta.get("duration") or 0.0) for meta in metadata]
    transitions_requested = any(
        _normalise_transition(item) != "cut" for item in transitions
    )

    first = metadata[0] if metadata else {}
    width = _even(first.get("width") or 832)
    height = _even(first.get("height") or 480)
    fps = min(max(int(first.get("fps") or 24), 1), 60)

    graph = build_xfade_filtergraph(
        durations,
        transitions,
        width=width,
        height=height,
        fps=fps,
        transition_seconds=getattr(settings, "FILM_TRANSITION_SECONDS", 1.0),
    )

    with tempfile.TemporaryDirectory(prefix="cineforge_film_") as tmpdir:
        tmp = Path(tmpdir)
        output = tmp / "final.mp4"

        if graph:
            cmd = [ffmpeg_bin, "-y"]
            for path in input_files:
                cmd += ["-i", str(path)]
            cmd += [
                "-filter_complex", graph,
                "-map", "[outv]",
                "-an",
                "-c:v", getattr(settings, "FFMPEG_CONCAT_TRANSCODING", "libx264"),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output),
            ]
        else:
            concat_list = tmp / "concat.txt"
            concat_list.write_text(
                "\n".join(f"file '{p.as_posix()}'" for p in input_files)
            )
            cmd = [
                ffmpeg_bin,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy",
                str(output),
            ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            msg = "Film assembly timed out."
            logger.error("Assembly timed out for project %s", project.id)
            project.assembly_error = msg
            project.save(update_fields=["assembly_error", "updated"])
            return {"success": False, "message": msg}
        except OSError as exc:
            msg = f"Could not run FFmpeg: {exc}"
            logger.error("Assembly OSError for project %s: %s", project.id, exc)
            project.assembly_error = msg
            project.save(update_fields=["assembly_error", "updated"])
            return {"success": False, "message": msg}

        if proc.returncode != 0:
            msg = f"FFmpeg assembly failed: {proc.stderr.strip()[-500:]}"
            logger.error("Assembly failed for project %s: %s", project.id, proc.stderr)
            project.assembly_error = msg
            project.save(update_fields=["assembly_error", "updated"])
            return {"success": False, "message": msg}

        if not output.exists() or output.stat().st_size < 1024:
            msg = "FFmpeg produced an empty or missing output file."
            project.assembly_error = msg
            project.save(update_fields=["assembly_error", "updated"])
            return {"success": False, "message": msg}

        # Move output into the project's assembled location
        dest_dir = settings.MEDIA_OUTPUTS_DIR / "films" / project.id.hex
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "final.mp4"
        if dest.exists():
            dest.unlink()
        try:
            shutil.move(str(output), str(dest))
        except OSError:
            shutil.copy2(str(output), str(dest))

        project.assembled_video.name = str(dest)
        project.assembly_error = ""
        project.save(update_fields=["assembled_video", "assembly_error", "updated"])

        logger.info("Assembled film %s -> %s", project.id, dest)
        if graph:
            message = (
                "Film assembled successfully with crossfade transitions "
                "(video only)."
            )
        elif transitions_requested:
            message = (
                "Film assembled successfully, but transitions were skipped "
                "because clip durations could not be probed."
            )
        else:
            message = "Film assembled successfully."
        return {"success": True, "message": message, "output": str(dest)}


# ---------------------------------------------------------------------------
# Scene workflow helpers
# ---------------------------------------------------------------------------


def next_scene_order(project: FilmProject) -> int:
    """Return the next free scene order for a project.

    Uses the maximum existing order rather than the scene count so deleting a
    scene can never cause a duplicate ``(project, order)`` value.
    """
    from django.db.models import Max

    current = project.scenes.aggregate(Max("order"))["order__max"]
    return 0 if current is None else current + 1


def move_scene(scene: FilmScene, direction: str) -> bool:
    """Move a scene one position up or down, returning True when it moved."""
    from django.db import transaction
    from django.db.models import Max

    if direction not in {"up", "down"}:
        return False

    project = scene.project
    with transaction.atomic():
        scenes = list(
            project.scenes.select_for_update().order_by("order", "created")
        )
        index = next((i for i, item in enumerate(scenes) if item.pk == scene.pk), None)
        if index is None:
            return False

        target = index - 1 if direction == "up" else index + 1
        if target < 0 or target >= len(scenes):
            return False

        current, other = scenes[index], scenes[target]
        current_order, other_order = current.order, other.order
        # Park the moving scene on a temporary order so the unique
        # (project, order) constraint is never violated mid-swap.
        temp_order = (project.scenes.aggregate(Max("order"))["order__max"] or 0) + 1000
        FilmScene.objects.filter(pk=current.pk).update(order=temp_order)
        FilmScene.objects.filter(pk=other.pk).update(order=current_order)
        FilmScene.objects.filter(pk=current.pk).update(order=other_order)

    return True


def generate_scene_clip(scene: FilmScene) -> tuple[bool, str | None]:
    """Create and queue a generation job for a film scene's prompt."""
    from django.conf import settings

    from generator.models import GenerationJob
    from generator.services import queue_generation_job, validate_and_normalize_job

    if scene.job_id and scene.job and not scene.job.is_terminal:
        return False, "A clip is already being generated for this scene."

    engine = (
        getattr(settings, "INFERENCE_DEFAULT_ADAPTER", "comfyui") or "comfyui"
    ).upper()
    if engine not in GenerationJob.Engine.values:
        engine = GenerationJob.Engine.COMFYUI

    prompt = scene.prompt
    if scene.character_reference_id and scene.character_reference:
        character = scene.character_reference
        prompt = (
            f"{prompt}\n\nCharacter: {character.name} — {character.description}"
        )

    job = GenerationJob(
        prompt=prompt,
        negative_prompt=scene.negative_prompt,
        engine=engine,
        duration_seconds=scene.duration_seconds,
    )
    ok, message, _warnings = validate_and_normalize_job(job)
    if not ok:
        return False, message or "The scene prompt is not valid."
    job.save()

    scene.job = job
    scene.status = FilmScene.Status.PENDING
    scene.save(update_fields=["job", "status", "updated"])

    queued, reason = queue_generation_job(job)
    if not queued:
        scene.status = FilmScene.Status.FAILED
        scene.save(update_fields=["status", "updated"])
        return False, reason or "The scene clip could not be queued."
    return True, reason


def sync_scene_from_job(job, video=None) -> int:
    """Copy a linked generation job's outcome onto its film scenes.

    Returns the number of scenes updated. Safe to call for jobs that are not
    linked to any scene.
    """
    from django.utils import timezone

    scenes = FilmScene.objects.filter(job=job)
    if not scenes.exists():
        return 0

    status_by_job = {
        "SUCCESS": FilmScene.Status.SUCCESS,
        "FAILED": FilmScene.Status.FAILED,
        "CANCELLED": FilmScene.Status.CANCELLED,
        "RUNNING": FilmScene.Status.RUNNING,
        "PENDING": FilmScene.Status.PENDING,
        "DEFERRED": FilmScene.Status.PENDING,
    }
    updates = {
        "status": status_by_job.get(job.status, FilmScene.Status.PENDING),
        "updated": timezone.now(),
    }

    if job.status == "SUCCESS" and job.output_file:
        updates["output_file"] = job.output_file.name
        if video is None:
            video = getattr(job, "generated_video", None)
        if video is not None:
            updates["source_video"] = video

    return scenes.update(**updates)

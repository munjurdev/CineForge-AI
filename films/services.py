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


def _xfade_unsupported(stderr: str | None) -> bool:
    """Return True when FFmpeg reports that the xfade filter is missing."""
    text = (stderr or "").lower()
    if "xfade" not in text:
        return False
    return "no such filter" in text or "error initializing" in text


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
    audio_present: list[bool] | None = None,
    force: bool = False,
) -> str | None:
    """Build an FFmpeg filter_complex graph that joins the clips.

    Returns ``None`` when every transition is a hard cut or when clip
    durations are unknown, so the caller can keep the fast concat path.
    ``force=True`` builds the graph even for an all-cut chain, which is needed
    when only some clips carry audio because stream copy cannot join those.

    The final video label is ``outv``. When ``audio_present`` holds at least
    one ``True``, a matching audio chain is built — concat for cuts and
    acrossfade for crossfades, using the same timings as the video — ending at
    ``outa``. Clips without an audio stream get silence of the right length so
    every scene stays in sync.
    """
    if len(durations) < 2:
        return None
    if any(duration <= 0 for duration in durations):
        return None

    cut_chain = [_normalise_transition(item) for item in transitions]
    cut_chain = cut_chain[: len(durations) - 1]
    while len(cut_chain) < len(durations) - 1:
        cut_chain.append("cut")

    if all(item == "cut" for item in cut_chain) and not force:
        return None

    has_audio = list(audio_present or [])
    include_audio = any(has_audio)

    parts: list[str] = []
    for index in range(len(durations)):
        parts.append(
            f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
            f"format=yuv420p,setpts=PTS-STARTPTS[v{index}]"
        )
        if not include_audio:
            continue
        clip_duration = durations[index]
        if index < len(has_audio) and has_audio[index]:
            parts.append(
                f"[{index}:a]aresample=48000,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"asetpts=PTS-STARTPTS,apad,atrim=0:{clip_duration:.3f}[a{index}]"
            )
        else:
            parts.append(
                f"anullsrc=channel_layout=stereo:sample_rate=48000,"
                f"atrim=0:{clip_duration:.3f},asetpts=PTS-STARTPTS[a{index}]"
            )

    current = "v0"
    current_audio = "a0"
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
            if include_audio:
                audio_label = f"ac{index}"
                parts.append(
                    f"[{current_audio}][a{index}]concat=n=2:v=0:a=1[{audio_label}]"
                )
                current_audio = audio_label
            elapsed += durations[index]
        else:
            offset = max(0.0, elapsed - duration)
            label = f"x{index}"
            parts.append(
                f"[{current}][v{index}]xfade=transition={transition}:"
                f"duration={duration:.3f}:offset={offset:.3f}[{label}]"
            )
            if include_audio:
                audio_label = f"ax{index}"
                parts.append(
                    f"[{current_audio}][a{index}]acrossfade=d={duration:.3f}:"
                    f"c1=tri:c2=tri[{audio_label}]"
                )
                current_audio = audio_label
            elapsed = elapsed + durations[index] - duration
        current = label

    parts.append(f"[{current}]null[outv]")
    if include_audio:
        parts.append(f"[{current_audio}]anull[outa]")
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

    # Scene clips may or may not carry the generated voiceover track; the
    # graph emits silence for the silent ones so the mix stays in sync.
    has_audio = [bool(meta.get("has_audio")) for meta in metadata]
    include_audio = any(has_audio)
    mixed_audio = include_audio and not all(has_audio)

    graph = build_xfade_filtergraph(
        durations,
        transitions,
        width=width,
        height=height,
        fps=fps,
        transition_seconds=getattr(settings, "FILM_TRANSITION_SECONDS", 1.0),
        audio_present=has_audio,
    )
    # Stream copy cannot join clips that disagree about having audio, so even an
    # all-cut film needs the filter graph in that case.
    if graph is None and mixed_audio:
        graph = build_xfade_filtergraph(
            durations,
            transitions,
            width=width,
            height=height,
            fps=fps,
            transition_seconds=getattr(settings, "FILM_TRANSITION_SECONDS", 1.0),
            audio_present=has_audio,
            force=True,
        )

    with tempfile.TemporaryDirectory(prefix="cineforge_film_") as tmpdir:
        tmp = Path(tmpdir)
        output = tmp / "final.mp4"

        def run_ffmpeg(command: list[str]):
            """Run FFmpeg with capped output capture."""
            return subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600,
            )

        def concat_command() -> list[str]:
            """Build the stream-copy concat command used for hard cuts."""
            concat_list = tmp / "concat.txt"
            concat_list.write_text(
                "\n".join(f"file '{p.as_posix()}'" for p in input_files)
            )
            return [
                ffmpeg_bin,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy",
                str(output),
            ]

        if graph:
            cmd = [ffmpeg_bin, "-y"]
            for path in input_files:
                cmd += ["-i", str(path)]
            video_codec = getattr(settings, "FFMPEG_CONCAT_TRANSCODING", "libx264")
            cmd += [
                "-filter_complex", graph,
                "-map", "[outv]",
            ]
            if include_audio and "[outa]" in graph:
                cmd += [
                    "-map", "[outa]",
                    "-c:a", "aac",
                    "-b:a", str(getattr(settings, "FILM_AUDIO_BITRATE", "192k")),
                ]
            else:
                # No scene carries audio: keep the render silent.
                cmd += ["-an"]
            cmd += ["-c:v", video_codec]
            # The xfade graph forces a re-encode; use a high-quality CRF so the
            # assembled film does not look worse than the source clips.
            if video_codec.startswith("libx26"):
                cmd += [
                    "-crf", str(getattr(settings, "FILM_VIDEO_CRF", 18)),
                    "-preset", "medium",
                ]
            cmd += [
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output),
            ]
        else:
            cmd = concat_command()

        transitions_unavailable = False
        try:
            proc = run_ffmpeg(cmd)
            if proc.returncode != 0 and graph and _xfade_unsupported(proc.stderr):
                logger.warning(
                    "FFmpeg build lacks xfade support; retrying with hard cuts."
                )
                proc = run_ffmpeg(concat_command())
                if proc.returncode == 0:
                    graph = None
                    transitions_unavailable = True
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
            audio_note = (
                "(scene audio kept)"
                if include_audio and "[outa]" in graph
                else "(video only)"
            )
            if transitions_requested:
                message = (
                    "Film assembled successfully with crossfade transitions "
                    f"{audio_note}."
                )
            else:
                message = (
                    f"Film assembled successfully with hard cuts {audio_note}."
                )
        elif transitions_unavailable:
            message = (
                "Film assembled successfully with hard cuts: the installed "
                "FFmpeg does not support xfade transitions (FFmpeg 4.3+ is "
                "required)."
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


def _normalise_engine(engine: str) -> str:
    """Map a configured default engine onto a real ``job.engine`` value.

    Legacy local engines are stored uppercase (``COMFYUI``, ``TEST``); cloud
    providers keep the lowercase key their provider spec registers. Anything
    unrecognised falls back to the local ComfyUI engine.
    """
    from generator.models import GenerationJob

    if (engine or "").upper() in GenerationJob.Engine.values:
        return engine.upper()

    key = (engine or "").strip().lower()
    try:
        from generator.providers import provider_keys

        if key in provider_keys():
            return key
    except Exception:  # pragma: no cover - registry unavailable
        pass
    return GenerationJob.Engine.COMFYUI


def generate_scene_clip(scene: FilmScene) -> tuple[bool, str | None]:
    """Create and queue a generation job for a film scene's prompt."""
    from django.conf import settings

    from generator.models import GenerationJob
    from generator.services import queue_generation_job, validate_and_normalize_job

    if scene.job_id and scene.job and not scene.job.is_terminal:
        return False, "A clip is already being generated for this scene."

    engine = _normalise_engine(
        getattr(settings, "INFERENCE_DEFAULT_ADAPTER", "comfyui") or "comfyui"
    )

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

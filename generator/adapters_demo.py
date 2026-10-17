"""
Test adapter — CPU-only, minimal video generation for CI and hardware-less
environments.

CineForge AI — Local AI Video Generation Platform.

This adapter does NOT claim to run a real AI video model. It creates a simple
animated clip (a colour gradient + optional text overlay) using ffmpeg so that
the rest of the application (jobs, library, film assembly, downloads, preview)
can be exercised end to end without a GPU.

It is intentionally small and predictable so tests can assert its behaviour.
"""
from __future__ import annotations

import logging
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from django.conf import settings

from .adapters import BaseAdapter, registry

logger = logging.getLogger("cineforge.generator.adapters_demo")


def _escape_drawtext(text: str) -> str:
    """Escape a string for FFmpeg's single-quoted drawtext expression."""
    return (
        text.replace("\\", "\\\\")
        .replace("'", r"\'")
        .replace(":", r"\:")
        .replace("%", r"\%")
    )


class TestAdapter(BaseAdapter):
    """Minimal CPU video adapter for development and testing."""

    name = "test"

    @property
    def can_run_locally(self) -> bool:
        return True

    @property
    def human_name(self) -> str:
        return "Test adapter (CPU demo)"

    @property
    def hardware_notes(self) -> list[str]:
        return [
            "Runs on CPU only.",
            "Produces a simple animated clip, not a real AI-generated video.",
            "Useful for testing pipeline, downloads, and film assembly.",
        ]

    def validate(self, job: Any) -> tuple[bool, str | None, list[str]]:
        w = job.width or 64
        h = job.height or 64
        if w < 16 or h < 16:
            return False, "Test adapter requires at least 16×16.", []
        if job.duration_seconds and job.duration_seconds <= 0:
            return False, "Duration must be positive.", []
        if job.fps and job.fps < 1:
            return False, "FPS must be at least 1.", []
        return True, None, []

    def generate(self, job: Any) -> str | Path | None:
        width = int(job.width or 64)
        height = int(job.height or 64)
        fps = int(job.fps or 12)
        duration = float(job.duration_seconds or 1.0)
        if duration <= 0:
            duration = 1.0

        prompt = (job.prompt or "").strip()
        if not prompt:
            prompt = "generated clip"

        frame_count = max(1, round(duration * fps))

        # Build an ffmpeg filter for a smooth colour sweep + optional text
        hue_start = secrets.randbelow(360)
        hue_end = (hue_start + 180) % 360

        with tempfile.TemporaryDirectory(prefix="cineforge_test_") as tmpdir:
            tmp = Path(tmpdir)
            output = tmp / "output.mp4"

            # Use the configured ffmpeg binary
            ffmpeg_bin = getattr(settings, "FFMPEG_BIN", "ffmpeg")
            if not shutil.which(ffmpeg_bin):
                logger.warning(
                    "ffmpeg not available; test adapter cannot generate video"
                )
                return None

            # Draw a colour-sweep background with a tidy grid hint + text.
            # Values are computed in Python because unquoted expressions with
            # commas (e.g. min(48,24)) break FFmpeg's filtergraph parser.
            font_size = max(10, min(height // 10, 24))
            label = _escape_drawtext(prompt[:60])
            hue = (
                f"hue=h='mod({hue_start}+t*{hue_end - hue_start}/max(1,{frame_count - 1}),360)':s=1"
            )
            drawtext = (
                f"drawtext=text='{label}':fontcolor=0xFFFFFF:fontsize={font_size}:"
                f"x=(w-text_w)/2:y=(h-text_h)/2:"
                f"enable='between(t,0.2,{max(0.4, duration - 0.2)})'"
            )
            grid = (
                f"drawgrid=width={max(width // 8, 4)}:height={max(height // 6, 4)}:"
                f"color=0x303030@0.25"
            )
            cmd = [
                ffmpeg_bin,
                "-y",
                "-f", "lavfi",
                "-i", f"color=c=0x101010:s={width}x{height}:r={fps}:d={duration}",
                "-filter_complex",
                f"{hue},{drawtext},{grid}",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-r", str(fps),
                "-movflags", "+faststart",
                "-t", str(duration),
                str(output),
            ]

            logger.debug("Test adapter ffmpeg cmd: %s", " ".join(cmd))
            try:
                p = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                logger.error("Test adapter timed out for job %s", job.id)
                return None
            except OSError as exc:
                logger.error("Test adapter could not run ffmpeg: %s", exc)
                return None

            if p.returncode != 0:
                logger.warning(
                    "Test adapter ffmpeg failed (rc=%s): %s",
                    p.returncode,
                    p.stderr.strip()[-500:],
                )
                # Some ffmpeg builds reject complex filter graphs with drawgrid;
                # fall back to a simpler color sweeper.
                cmd_simple = [
                    ffmpeg_bin,
                    "-y",
                    "-f", "lavfi",
                    "-i", f"color=c=0x101010:s={width}x{height}:r={fps}:d={duration}",
                    "-vf",
                    f"{hue},{drawtext}",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-r", str(fps),
                    "-movflags", "+faststart",
                    "-t", str(duration),
                    str(output),
                ]
                try:
                    p2 = subprocess.run(
                        cmd_simple,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=120,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Test adapter fallback failed: %s", exc)
                    return None
                if p2.returncode != 0:
                    logger.error(
                        "Test adapter fallback failed (rc=%s): %s",
                        p2.returncode,
                        p2.stderr.strip()[-500:],
                    )
                    return None
                if not output.exists():
                    return None

            if not output.exists() or output.stat().st_size < 1024:
                logger.warning("Test adapter produced too-small output: %s", output)
                return None

            # Copy to a persistent location under media/temp so the service layer
            # can move it into outputs. Return the path to that copy.
            dest_parent = settings.MEDIA_TEMP_DIR / job.id.hex
            dest_parent.mkdir(parents=True, exist_ok=True)
            dest = dest_parent / "test_output.mp4"

            # Remove any previous file
            if dest.exists():
                dest.unlink()

            shutil.copy2(str(output), str(dest))
            logger.info("Test adapter wrote %s bytes to %s", dest.stat().st_size, dest)
            return str(dest)


registry.register(TestAdapter.name, TestAdapter)

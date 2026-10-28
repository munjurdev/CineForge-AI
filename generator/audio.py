"""
Automatic AI voiceover for generated clips.

CineForge AI — Local AI Video Generation Platform.

The video models CineForge drives (Wan 2.x, LTX) only produce silent frames, so
the spoken track is generated from the job's prompt with `edge-tts` (free
Microsoft neural voices, no API key) and muxed into the clip with FFmpeg.

Everything here is best-effort: if FFmpeg or edge-tts is missing, or the
network is unreachable, the clip is returned untouched and the job still
succeeds. A missing voice track must never fail a generation.
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from django.conf import settings

logger = logging.getLogger("cineforge.generator.audio")

# Words that describe the look of a shot rather than what happens on screen.
# Spoken aloud they turn narration into gibberish, so they are trimmed from the
# end of the prompt before synthesis.
_STYLE_TAIL = re.compile(
    r"(?:^|\s*,\s*|\s+)(?:"
    r"(?:\d+\s*k|4k|8k|1080p|720p|hd|uhd|"
    r"cinematic|cinematic lighting|photorealistic|hyperrealistic|realistic|"
    r"ultra[- ]detailed|highly detailed|detailed|sharp focus|shallow depth of field|"
    r"depth of field|bokeh|film grain|hdr|"
    r"unreal engine(?:\s*\d+)?|octane render|ray ?tracing|"
    r"masterpiece|best quality|high quality|award[- ]winning|"
    r"8k resolution|wide angle|close[- ]up shot|slow motion|"
    r"vertical video|horizontal video)\b"
    r")+\s*$",
    re.IGNORECASE,
)

# Bracketed camera/stage directions: "(camera pans left)", "[cut to wide]".
_DIRECTIONS = re.compile(r"[\(\[\{][^\)\]\}]{0,120}[\)\]\}]")

# Sentence punctuation that reads well but is not useful in a spoken line.
_TRAILING_PUNCT = " ,;:-–—."


def narration_text_for(prompt: str | None, max_chars: int | None = None) -> str:
    """Turn a generation prompt into a short line worth speaking aloud.

    Removes camera directions and trailing style keywords ("4k, cinematic,
    hyperrealistic"), collapses whitespace, and cuts the result at a word
    boundary. Returns an empty string when nothing speakable is left.
    """
    text = _DIRECTIONS.sub(" ", prompt or "")
    text = " ".join(text.split())

    previous = None
    while text and text != previous:
        previous = text
        text = _STYLE_TAIL.sub("", text).strip(_TRAILING_PUNCT)

    if not text:
        return ""

    limit = max_chars or int(getattr(settings, "VOICEOVER_MAX_CHARS", 240))
    if len(text) > limit:
        cut = text[:limit]
        boundary = cut.rfind(" ")
        if boundary > limit // 2:
            cut = cut[:boundary]
        text = cut.rstrip(_TRAILING_PUNCT) + "…"

    return text


def _ffmpeg_bin() -> str | None:
    """Return a usable FFmpeg binary path, or None when it is not installed."""
    configured = str(getattr(settings, "FFMPEG_BIN", "ffmpeg"))
    if shutil.which(configured):
        return configured
    return None


def _text_to_speech_sync(text: str, dest: Path, *, voice: str, rate: str, volume: str) -> bool:
    """Synthesise `text` into `dest` with edge-tts (blocking)."""
    try:
        import edge_tts
    except ImportError:
        logger.warning(
            "edge-tts is not installed; skipping the voiceover. "
            "Install it with: pip install edge-tts"
        )
        return False

    async def _run() -> None:
        communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
        await communicate.save(str(dest))

    try:
        asyncio.run(_run())
    except Exception as exc:  # network, TLS, voice name, event loop
        logger.warning("Could not synthesise the voiceover with edge-tts: %s", exc)
        return False

    return dest.exists() and dest.stat().st_size > 512


def synthesize_speech(text: str, dest: Path) -> bool:
    """Generate a speech MP3 for `text` at `dest`. Returns True on success."""
    text = (text or "").strip()
    if not text:
        return False

    voice = str(getattr(settings, "VOICEOVER_VOICE", "bn-BD-NabanitaNeural"))
    rate = str(getattr(settings, "VOICEOVER_RATE", "+0%"))
    volume = str(getattr(settings, "VOICEOVER_VOLUME", "+0%"))

    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = _text_to_speech_sync(text, dest, voice=voice, rate=rate, volume=volume)
    if ok:
        logger.info(
            "Voiceover synthesised (%s, %s chars) -> %s", voice, len(text), dest.name
        )
    return ok


def mux_audio(video: Path, audio: Path, dest: Path) -> Path | None:
    """Mux `audio` into `video`, writing `dest`.

    The video stream is copied, not re-encoded, so muxing never costs quality.
    Audio is padded to the video length so the clip keeps its full duration.
    """
    ffmpeg = _ffmpeg_bin()
    if ffmpeg is None:
        logger.warning("FFmpeg not found; cannot mux the voiceover into the clip.")
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    bitrate = str(getattr(settings, "VOICEOVER_AUDIO_BITRATE", "192k"))

    def build(copy_video: bool) -> list[str]:
        command = [
            ffmpeg,
            "-y",
            "-i", str(video),
            "-i", str(audio),
            "-map", "0:v:0",
            "-map", "1:a:0",
        ]
        if copy_video:
            command += ["-c:v", "copy"]
        else:
            command += ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
        command += [
            "-c:a", "aac",
            "-b:a", bitrate,
            "-af", "apad",
            "-shortest",
            "-movflags", "+faststart",
            str(dest),
        ]
        return command

    # Stream copy only works when the container can hold the source codec
    # (H.264 in MP4). Fall back to a high-quality re-encode otherwise.
    for copy_video in (True, False):
        proc = subprocess.run(
            build(copy_video),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
        )
        if proc.returncode == 0 and dest.exists() and dest.stat().st_size > 1024:
            logger.info("Muxed voiceover into %s (video %s)", dest.name,
                        "copied" if copy_video else "re-encoded")
            return dest
        logger.warning(
            "FFmpeg mux failed (copy=%s, rc=%s): %s",
            copy_video,
            proc.returncode,
            (proc.stderr or "").strip()[-300:],
        )

    return None


def add_voiceover_to_video(job: Any, video_path: Path) -> Path:
    """Add an AI-spoken narration track to `video_path`.

    Returns the final path: the muxed file when everything worked, otherwise the
    original clip. Never raises — a voiceover problem must not fail a job.
    """
    if not getattr(settings, "VOICEOVER_ENABLED", True):
        return video_path
    if not video_path.exists():
        return video_path

    text = narration_text_for(getattr(job, "prompt", "") or "")
    if not text:
        logger.info("Job %s has nothing speakable in its prompt; keeping it silent.", getattr(job, "id", "?"))
        return video_path

    try:
        with tempfile.TemporaryDirectory(prefix="cineforge_voice_") as tmp:
            audio = Path(tmp) / "voiceover.mp3"
            if not synthesize_speech(text, audio):
                return video_path

            dest = video_path.with_name(f"{video_path.stem}_voiced.mp4")
            muxed = mux_audio(video_path, audio, dest)
            if muxed is None:
                return video_path

            try:
                video_path.unlink()
            except OSError:  # pragma: no cover - file lock on Windows
                logger.debug("Could not remove the silent intermediate %s", video_path)
            return muxed
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Voiceover step failed for job %s: %s", getattr(job, "id", "?"), exc)
        return video_path

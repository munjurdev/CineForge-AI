"""
System diagnostics service — detect hardware and backend availability.

CineForge AI — Local AI Video Generation Platform.

These checks are best-effort and graceful: when something is unavailable
(for example CUDA or a GPU), the page should still render useful information
instead of crashing.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from django.conf import settings

logger = logging.getLogger("cineforge.system_monitor")


def _which(name: str) -> str | None:
    return shutil.which(name)


def _disk_usage(path: Path) -> dict[str, Any]:
    """Return free/total disk space for the given path, or empty dict on failure."""
    try:
        import shutil as _shutil
        usage = _shutil.disk_usage(path)
        return {
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
        }
    except OSError:
        return {}


def detect_system_info() -> dict[str, Any]:
    """Return a dict of system diagnostics for the frontend."""
    result: dict[str, Any] = {
        "os": os.name,
        "platform": getattr(os, "uname", lambda: {})() if hasattr(os, "uname") else {},
        "python": {
            "executable": shutil.which("python") or shutil.which("python3") or "",
            "version": getattr(os, "python_version", lambda: "unknown")() if hasattr(os, "python_version") else "unknown",
        },
        "memory": {},
        "gpu": {},
        "disk": {},
        "tools": {},
        "model_suggestions": [],
        "warnings": [],
    }

    # RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        result["memory"] = {
            "total": mem.total,
            "available": mem.available,
            "percent": mem.percent,
        }
    except Exception:
        result["memory"] = {}

    # GPU / CUDA
    try:
        import torch
        result["gpu"] = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1) if torch.cuda.is_available() else None,
        }
    except Exception:
        result["gpu"] = {"cuda_available": False}

    # Tools — honour the configured binary first (FFMPEG_BIN may point at a
    # bundled build), then fall back to a PATH lookup.
    configured_ffmpeg = getattr(settings, "FFMPEG_BIN", "ffmpeg")
    ffmpeg = _which(configured_ffmpeg) or _which("ffmpeg")
    configured_ffprobe = str(
        Path(configured_ffmpeg).with_name(
            Path(configured_ffmpeg).name.replace("ffmpeg", "ffprobe", 1)
        )
    )
    ffprobe = _which(configured_ffprobe) or _which("ffprobe") or ffmpeg
    result["tools"] = {
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "comfyui_running": False,
    }

    if ffmpeg:
        # ffmpeg/ffprobe accept "-version"; "--version" exits non-zero and
        # would leave this blank.
        result["tools"]["ffmpeg_version"] = _tool_version(ffmpeg, ["-version"])
    else:
        result["warnings"].append("FFmpeg is not installed or not on PATH.")

    # ComfyUI reachability (best-effort)
    try:
        import httpx
        base = getattr(settings, "COMFYUI_BASE_URL", "http://127.0.0.1:8188")
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base}/system_stats")
            result["tools"]["comfyui_running"] = resp.status_code == 200
    except Exception:
        result["tools"]["comfyui_running"] = False

    # Disk
    result["disk"] = _disk_usage(Path(settings.MEDIA_ROOT))

    # Model suggestions based on hardware (conservative)
    vram_mb = result["gpu"].get("vram_total_mb")
    if isinstance(vram_mb, (int, float)) and vram_mb >= 8000:
        result["model_suggestions"] = [
            "Wan 2.1 (7B/14B) — good quality, higher VRAM requirement",
            "LTX-Video — good for shorter clips, moderate VRAM",
        ]
    elif isinstance(vram_mb, (int, float)) and vram_mb >= 4000:
        result["model_suggestions"] = [
            "LTX-Video — moderate VRAM, good for shorter clips",
            "Wan 2.1 (quantized) — reduced VRAM with some quality trade-off",
        ]
    elif isinstance(vram_mb, (int, float)) and vram_mb >= 0:
        result["model_suggestions"] = [
            "CPU-only demo via the Test adapter",
            "Consider a cloud backend if local VRAM is insufficient",
        ]
    else:
        result["model_suggestions"] = [
            "CPU-only demo via the Test adapter",
        ]
        result["warnings"].append("GPU/VRAM could not be detected.")

    # Warnings for missing critical components
    if not result["tools"]["ffmpeg"]:
        result["warnings"].append("Install FFmpeg to enable video assembly and probing.")
    if not result["tools"]["comfyui_running"]:
        result["warnings"].append("ComfyUI is not running or not reachable at the configured URL.")

    return result


def get_system_summary() -> dict[str, Any]:
    """Return a compact hardware/backend summary for the dashboard."""
    info = detect_system_info()
    gpu = info.get("gpu", {})
    memory = info.get("memory", {})
    tools = info.get("tools", {})
    return {
        "has_gpu": bool(gpu.get("cuda_available")),
        "gpu_name": gpu.get("device_name"),
        "vram_mb": gpu.get("vram_total_mb"),
        "ram_available": memory.get("available"),
        "ffmpeg_available": bool(tools.get("ffmpeg")),
        "comfyui_running": bool(tools.get("comfyui_running")),
        "warnings": info.get("warnings", []),
    }


def _tool_version(bin: str, args: list[str]) -> str | None:
    try:
        proc = subprocess.run(
            [bin] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
            return line[:120]
    except Exception:
        pass
    return None

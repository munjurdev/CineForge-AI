"""
Luma Dream Machine (Ray 2) — natural, lifelike motion at a low price.

    POST {base}/dream-machine/v1/generations -> {"id": "…"}
    GET  {base}/dream-machine/v1/generations/{id} -> state + assets.video

Note on image-to-video: Luma only accepts *hosted* (public CDN) image URLs as
keyframes, and CineForge stores reference images privately, so this adapter
declares text-to-video only.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import httpx

from .base import CloudVideoAdapter, PollResult, PollState
from .registry import register_provider
from .spec import CAP_ASPECT, CAP_RESOLUTION, ProviderSpec, read_env

DEFAULT_BASE_URL = "https://api.lumalabs.ai"

# Ray 2 accepts these clip lengths, as strings with the unit attached.
_DURATION_CHOICES = ("5s", "9s")
_RESOLUTIONS = (("540p", 640), ("720p", 1280), ("1080p", 1920), ("4k", 3840))

SPEC = ProviderSpec(
    key="luma",
    label="Luma Ray 2",
    kind="cloud",
    order=30,
    env_keys=("LUMA_API_KEY",),
    optional_env=("LUMA_API_BASE", "LUMA_MODEL"),
    config_hint="Create a key at https://platform.lumalabs.ai/ and set LUMA_API_KEY.",
    capabilities=frozenset({CAP_ASPECT, CAP_RESOLUTION}),
    max_duration=9.0,
    default_model="ray-2",
    preset={"aspect_ratio": "16:9", "duration_seconds": 5.0},
    website="https://lumalabs.ai/",
    docs_url="https://docs.lumalabs.ai/docs/video-generation",
    pricing="credit based; check lumalabs.ai/api for current per-render credits",
    estimated_seconds=60.0,
)


class LumaAdapter(CloudVideoAdapter):
    """Luma Dream Machine generations."""

    name = "luma"

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {read_env('LUMA_API_KEY')}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @property
    def _base_url(self) -> str:
        return read_env("LUMA_API_BASE", DEFAULT_BASE_URL).rstrip("/")

    @staticmethod
    def _duration(job: Any) -> str:
        seconds = float(getattr(job, "duration_seconds", 5) or 5)
        return _DURATION_CHOICES[1] if seconds > 6 else _DURATION_CHOICES[0]

    @staticmethod
    def _resolution(job: Any) -> str:
        width = int(getattr(job, "width", 0) or 0)
        for label, threshold in reversed(_RESOLUTIONS):
            if width >= threshold:
                return label
        return "720p"

    def build_payload(self, job: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": getattr(job, "prompt", ""),
            "model": read_env("LUMA_MODEL", SPEC.default_model),
            "duration": self._duration(job),
            "resolution": self._resolution(job),
        }
        ratio = (getattr(job, "aspect_ratio", "") or "").strip()
        if ratio:
            payload["aspect_ratio"] = ratio
        return payload

    def submit(self, client: httpx.Client, job: Any) -> str:
        response = client.post(
            f"{self._base_url}/dream-machine/v1/generations",
            json=self.build_payload(job),
        )
        response.raise_for_status()
        remote_id = (response.json() or {}).get("id")
        if not remote_id:
            raise RuntimeError(f"Luma returned no generation id: {response.text[:300]}")
        return str(remote_id)

    def poll(self, client: httpx.Client, remote_id: str) -> PollResult:
        response = client.get(f"{self._base_url}/dream-machine/v1/generations/{remote_id}")
        response.raise_for_status()
        payload = response.json() or {}

        state = str(payload.get("state", "")).lower()
        if state == "completed":
            video = ((payload.get("assets") or {}).get("video")) or None
            if not video:
                return PollResult(
                    state=PollState.FAILED,
                    error=f"Luma completed without a video: {str(payload)[:300]}",
                )
            return PollResult(state=PollState.SUCCEEDED, video_url=str(video), raw=payload)
        if state == "failed":
            reason = payload.get("failure_reason") or "generation failed"
            return PollResult(state=PollState.FAILED, error=str(reason), raw=payload)
        if state == "cancelled":
            return PollResult(state=PollState.CANCELLED, raw=payload)

        label = {"queued": "queued", "dreaming": "dreaming"}.get(state, state or "queued")
        return PollResult(
            state=PollState.PENDING if state == "queued" else PollState.RUNNING,
            message=f"Luma is {label}…",
            raw=payload,
        )


register_provider(replace(SPEC, adapter_cls=LumaAdapter))

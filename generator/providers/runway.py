"""
Runway — director-grade camera control and the strongest image-to-video of
the hosted providers.

    POST {base}/v1/text_to_video | /v1/image_to_video   -> {"id": "…"}
    GET  {base}/v1/tasks/{id}                           -> status + output[0]

Auth is a bearer token plus the mandatory ``X-Runway-Version`` header.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import httpx

from .base import CloudVideoAdapter, PollResult, PollState
from .registry import register_provider
from .spec import CAP_ASPECT, CAP_GUIDANCE, CAP_I2V, CAP_SEED, CAP_STEPS, ProviderSpec, read_env

DEFAULT_BASE_URL = "https://api.dev.runwayml.com"
DEFAULT_API_VERSION = "2024-11-06"

# Runway accepts an explicit list of ratios rather than arbitrary widths.
_RUNWAY_RATIOS = ("1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672")

SPEC = ProviderSpec(
    key="runway",
    label="Runway Gen-4",
    kind="cloud",
    order=20,
    env_keys=("RUNWAY_API_KEY",),
    optional_env=("RUNWAY_API_BASE", "RUNWAY_API_VERSION", "RUNWAY_MODEL"),
    config_hint="Create an API key at https://dev.runway.com/ and set RUNWAY_API_KEY.",
    capabilities=frozenset({CAP_I2V, CAP_ASPECT}),
    max_duration=10.0,
    default_model="gen4.5",
    preset={"aspect_ratio": "16:9", "duration_seconds": 5.0},
    website="https://runwayml.com/",
    docs_url="https://docs.dev.runwayml.com/",
    pricing="billed in Runway credits (about 12 credits per second)",
    estimated_seconds=90.0,
)


class RunwayAdapter(CloudVideoAdapter):
    """Runway text/image-to-video tasks."""

    name = "runway"

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {read_env('RUNWAY_API_KEY')}",
            "X-Runway-Version": read_env("RUNWAY_API_VERSION", DEFAULT_API_VERSION),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @property
    def _base_url(self) -> str:
        return read_env("RUNWAY_API_BASE", DEFAULT_BASE_URL).rstrip("/")

    def _ratio(self, job: Any) -> str:
        """Map the job's aspect ratio onto a ratio Runway accepts."""
        ratio = (getattr(job, "aspect_ratio", "") or "").strip()
        width = int(getattr(job, "width", 0) or 0)
        height = int(getattr(job, "height", 0) or 0)
        if width and height:
            if width > height * 1.15:
                candidates = ("1280:720", "1584:672", "1104:832")
            elif height > width * 1.15:
                candidates = ("720:1280", "832:1104")
            else:
                candidates = ("960:960",)
        else:
            candidates = _RUNWAY_RATIOS
        for candidate in candidates:
            if candidate in _RUNWAY_RATIOS:
                return candidate
        if ratio in _RUNWAY_RATIOS:
            return ratio
        return "1280:720"

    def build_payload(self, job: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": read_env("RUNWAY_MODEL", SPEC.default_model),
            "promptText": getattr(job, "prompt", "")[:500],
            "ratio": self._ratio(job),
            "duration": int(min(10, max(1, round(float(job.duration_seconds or 5))))),
        }
        image = self.reference_image(job)
        if image:
            data, mime = image
            import base64

            payload["promptImage"] = (
                f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
            )
        return payload

    def submit(self, client: httpx.Client, job: Any) -> str:
        endpoint = "image_to_video" if getattr(job, "reference_image", None) else "text_to_video"
        response = client.post(f"{self._base_url}/v1/{endpoint}", json=self.build_payload(job))
        response.raise_for_status()
        remote_id = (response.json() or {}).get("id")
        if not remote_id:
            raise RuntimeError(f"Runway returned no task id: {response.text[:300]}")
        return str(remote_id)

    def poll(self, client: httpx.Client, remote_id: str) -> PollResult:
        response = client.get(f"{self._base_url}/v1/tasks/{remote_id}")
        response.raise_for_status()
        payload = response.json() or {}

        status = str(payload.get("status", "")).upper()
        failure = payload.get("failure") or payload.get("failureCode") or ""
        if status in {"FAILED", "CANCELLED"}:
            return PollResult(
                state=PollState.FAILED if status == "FAILED" else PollState.CANCELLED,
                error=str(failure) or f"Runway task {status.lower()}",
                raw=payload,
            )
        if status == "SUCCEEDED":
            output = payload.get("output") or []
            video = output[0] if isinstance(output, list) and output else None
            if not video:
                return PollResult(
                    state=PollState.FAILED,
                    error="Runway succeeded but returned no output URL.",
                    raw=payload,
                )
            return PollResult(state=PollState.SUCCEEDED, video_url=str(video), raw=payload)

        return PollResult(
            state=PollState.RUNNING if status else PollState.PENDING,
            message=f"Runway task {status.lower() or 'queued'}…",
            raw=payload,
        )


register_provider(replace(SPEC, adapter_cls=RunwayAdapter))

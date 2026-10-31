"""
Google Veo via the Gemini API (the model behind Google Flow).

Flow is the consumer UI; Veo is the model, and this adapter talks to it
directly through the Gemini API's long-running operations:

    POST {base}/models/{model}:predictLongRunning   -> {"name": "operations/…"}
    GET  {base}/{operation name}                    -> done + video uri
    GET  the video uri (with the API key)           -> MP4 bytes

Model ids move quickly — Google retired the Veo 3.1 *preview* models in
October 2026 — so ``VEO_MODEL`` must be set to a model your account can call.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

import base64
from dataclasses import replace
from typing import Any

import httpx

from .base import CloudVideoAdapter, PollResult, PollState
from .registry import register_provider
from .spec import CAP_ASPECT, CAP_AUDIO, CAP_I2V, ProviderSpec, read_env

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

SPEC = ProviderSpec(
    key="veo",
    label="Google Veo",
    kind="cloud",
    order=10,
    env_keys=("VEO_API_KEY", "VEO_MODEL"),
    optional_env=("VEO_BASE_URL",),
    config_hint=(
        "Create a key at https://aistudio.google.com/apikey and set "
        "VEO_MODEL to a Veo model your account can call (the 3.1 preview "
        "models were retired in October 2026 — check the Gemini API video "
        "docs for the current id)."
    ),
    capabilities=frozenset({CAP_I2V, CAP_AUDIO, CAP_ASPECT}),
    max_duration=8.0,
    default_model="",
    preset={"aspect_ratio": "16:9", "duration_seconds": 8.0},
    website="https://deepmind.google/technologies/veo/",
    docs_url="https://ai.google.dev/gemini-api/docs/video",
    pricing="~$0.05-0.60 per generated second depending on tier and resolution",
    estimated_seconds=75.0,
)


class VeoAdapter(CloudVideoAdapter):
    """Veo text/image-to-video through the Gemini API."""

    name = "veo"

    # --------------------------------------------------------------- requests
    def headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": read_env("VEO_API_KEY"),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @property
    def _base_url(self) -> str:
        return read_env("VEO_BASE_URL", DEFAULT_BASE_URL).rstrip("/")

    @property
    def _model(self) -> str:
        return read_env("VEO_MODEL").strip()

    def _operation_url(self, operation_name: str) -> str:
        # The API returns "operations/<id>" (older builds return the fully
        # qualified "models/<id>/operations/<id>"); both are addressable
        # relative to the version base.
        if operation_name.startswith("models/"):
            return f"{self._base_url}/{operation_name}"
        return (
            f"{self._base_url}/models/{self._model}/operations/"
            f"{operation_name.split('/')[-1]}"
        )

    def build_payload(self, job: Any) -> dict[str, Any]:
        instance: dict[str, Any] = {"prompt": getattr(job, "prompt", "")}
        image = self.reference_image(job)
        if image:
            payload, mime = image
            instance["imageFile"] = {
                "bytes": base64.b64encode(payload).decode("ascii"),
                "mimeType": mime,
            }

        parameters: dict[str, Any] = {}
        ratio = getattr(job, "aspect_ratio", "") or ""
        if ratio:
            parameters["aspectRatio"] = ratio
        duration = getattr(job, "duration_seconds", None)
        if duration:
            parameters["durationSeconds"] = int(round(duration))
        return {"instances": [instance], "parameters": parameters}

    def submit(self, client: httpx.Client, job: Any) -> str:
        url = f"{self._base_url}/models/{self._model}:predictLongRunning"
        response = client.post(url, json=self.build_payload(job))
        response.raise_for_status()
        operation = (response.json() or {}).get("name")
        if not operation:
            raise RuntimeError(f"Veo returned no operation id: {response.text[:300]}")
        return operation

    # ------------------------------------------------------------------ polls
    def poll(self, client: httpx.Client, remote_id: str) -> PollResult:
        response = client.get(self._operation_url(remote_id))
        response.raise_for_status()
        payload = response.json() or {}

        error = payload.get("error")
        if error:
            return PollResult(
                state=PollState.FAILED,
                error=str(error.get("message") or error),
            )

        response_body = payload.get("response") or {}
        if not payload.get("done"):
            return PollResult(
                state=PollState.RUNNING,
                message="Veo is rendering…",
                raw=payload,
            )

        video = self._extract_video_uri(response_body)
        if not video:
            return PollResult(
                state=PollState.FAILED,
                error=f"Veo finished without a video: {str(payload)[:300]}",
            )
        return PollResult(state=PollState.SUCCEEDED, video_url=video, raw=payload)

    @staticmethod
    def _extract_video_uri(body: dict[str, Any]) -> str | None:
        """Walk the (version-dependent) response shape for a video URI."""
        samples = body.get("generatedSamples") or body.get("videos") or []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            candidate = (
                sample.get("video")
                or sample.get("generatedVideo")
                or sample
            )
            if isinstance(candidate, dict) and candidate.get("uri"):
                return str(candidate["uri"])
            if isinstance(candidate, str) and candidate.startswith("http"):
                return candidate
        if isinstance(body.get("uri"), str):
            return str(body["uri"])
        return None

    def download_headers(self) -> dict[str, str]:
        # The generated file lives behind the same API key.
        return {"x-goog-api-key": read_env("VEO_API_KEY")}


register_provider(replace(SPEC, adapter_cls=VeoAdapter))

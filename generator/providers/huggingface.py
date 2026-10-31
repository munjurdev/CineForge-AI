"""
Hugging Face Inference Providers — a gateway to open video models
(Wan, LTX-Video, HunyuanVideo…) hosted by partner providers, billed at the
provider's own rate.

Unlike every other provider here the call is **synchronous**: one POST
returns the encoded video bytes, so this adapter sets ``synchronous = True``
and implements ``run_sync()`` instead of submit/poll.

    POST {base}/{provider}/models/{model} -> video bytes (or a JSON error)

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import httpx

from .base import CloudVideoAdapter, PollResult, PollState
from .registry import register_provider
from .spec import CAP_GUIDANCE, CAP_NEGATIVE, CAP_SEED, CAP_STEPS, ProviderSpec, read_env

DEFAULT_BASE_URL = "https://router.huggingface.co"

SPEC = ProviderSpec(
    key="huggingface",
    label="Hugging Face",
    kind="cloud",
    order=60,
    env_keys=("HF_TOKEN", "HF_MODEL"),
    optional_env=("HF_API_BASE", "HF_PROVIDER"),
    config_hint=(
        "Create a token with Inference Providers access at "
        "https://huggingface.co/settings/tokens and set HF_TOKEN. Set HF_MODEL "
        "to a text-to-video model, for example Wan-AI/Wan2.2-TI2V-5B."
    ),
    # The documented payload for this task accepts all four tuning fields.
    capabilities=frozenset({CAP_SEED, CAP_STEPS, CAP_GUIDANCE, CAP_NEGATIVE}),
    default_model="Wan-AI/Wan2.2-TI2V-5B",
    preset={"duration_seconds": 5.0, "fps": 16},
    website="https://huggingface.co/",
    docs_url="https://huggingface.co/docs/inference-providers/en/tasks/text-to-video",
    pricing="pass-through: you pay the underlying provider's per-second rate",
    estimated_seconds=180.0,
)


class HuggingFaceAdapter(CloudVideoAdapter):
    """Synchronous text-to-video through the Hugging Face router."""

    name = "huggingface"
    synchronous = True

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {read_env('HF_TOKEN')}",
            "Content-Type": "application/json",
            "Accept": "video/mp4, application/octet-stream, application/json",
        }

    @property
    def _url(self) -> str:
        base = read_env("HF_API_BASE", DEFAULT_BASE_URL).rstrip("/")
        provider = read_env("HF_PROVIDER", "hf-inference").strip("/")
        model = read_env("HF_MODEL").strip()
        return f"{base}/{provider}/models/{model}"

    def build_payload(self, job: Any) -> dict[str, Any]:
        parameters: dict[str, Any] = {}

        steps = getattr(job, "steps", None)
        if steps:
            parameters["num_inference_steps"] = int(steps)

        guidance = getattr(job, "guidance_scale", None)
        if guidance:
            parameters["guidance_scale"] = float(guidance)

        seed = getattr(job, "seed", None)
        if seed is not None:
            parameters["seed"] = int(seed)

        # The task has no duration field; clip length is driven by frames.
        duration = float(getattr(job, "duration_seconds", 0) or 0)
        fps = int(getattr(job, "fps", 0) or 0)
        if duration and fps:
            parameters["num_frames"] = max(1, min(1000, round(duration * fps)))

        negative = getattr(job, "negative_prompt", "") or ""
        if negative and negative != self._default_negative_prompt():
            parameters["negative_prompt"] = [negative]

        payload: dict[str, Any] = {"inputs": getattr(job, "prompt", "")}
        if parameters:
            payload["parameters"] = parameters
        return payload

    def run_sync(self, client: httpx.Client, job: Any) -> bytes | str:
        from ..services import job_set_progress

        job_set_progress(job, 0.1, f"{self.spec.label} is generating the video…")
        response = client.post(
            self._url,
            json=self.build_payload(job),
            timeout=float(read_env("VIDEO_PROVIDER_SYNC_TIMEOUT", "600")),
        )
        content_type = response.headers.get("content-type", "")
        if response.status_code >= 400 or "application/json" in content_type:
            # Errors come back as JSON even on a 200.
            detail = response.text[:400]
            try:
                body = response.json()
                detail = str(body.get("error") or body.get("message") or detail)
            except ValueError:
                pass
            raise RuntimeError(
                f"Hugging Face request failed (HTTP {response.status_code}): {detail}"
            )

        payload = response.content
        if "application/json" in content_type:
            raise RuntimeError(f"Unexpected JSON response: {response.text[:300]}")
        return payload

    # Not used by the synchronous path, but the base class requires it.
    def submit(self, client: httpx.Client, job: Any) -> str:  # pragma: no cover
        raise NotImplementedError("Hugging Face responds synchronously.")

    def poll(self, client: httpx.Client, remote_id: str) -> PollResult:  # pragma: no cover
        raise NotImplementedError("Hugging Face responds synchronously.")


register_provider(replace(SPEC, adapter_cls=HuggingFaceAdapter))

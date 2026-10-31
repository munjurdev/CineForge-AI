"""
Replicate — one API key, hundreds of open video models (Wan, LTX, Hunyuan…).

    POST {base}/v1/models/{owner}/{name}/predictions -> {"id": "…"}
    GET  {base}/v1/predictions/{id}                  -> status + output

The ``input`` schema is per model, so the adapter sends a sensible default
(``prompt``) plus whatever ``REPLICATE_EXTRA_INPUT`` declares as JSON — that
lets a job target any model on the platform without a code change.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import httpx

from .base import CloudVideoAdapter, PollResult, PollState
from .registry import register_provider
from .spec import ProviderSpec, read_env

DEFAULT_BASE_URL = "https://api.replicate.com"

SPEC = ProviderSpec(
    key="replicate",
    label="Replicate",
    kind="cloud",
    order=50,
    env_keys=("REPLICATE_API_TOKEN", "REPLICATE_MODEL"),
    optional_env=("REPLICATE_API_BASE", "REPLICATE_EXTRA_INPUT"),
    config_hint=(
        "Get a token at https://account.replicate.com/tokens and set "
        "REPLICATE_API_TOKEN plus REPLICATE_MODEL (for example "
        "'wan-video/wan-2.2-t2v-fast'). Optional: REPLICATE_EXTRA_INPUT as a "
        "JSON object of extra input fields for that model."
    ),
    capabilities=frozenset(),  # input schema depends entirely on the model
    default_model="",
    website="https://replicate.com/",
    docs_url="https://docs.replicate.com/",
    pricing="per prediction, varies by model (usually a few cents per clip)",
    estimated_seconds=120.0,
)


class ReplicateAdapter(CloudVideoAdapter):
    """Replicate predictions."""

    name = "replicate"

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {read_env('REPLICATE_API_TOKEN')}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @property
    def _base_url(self) -> str:
        return read_env("REPLICATE_API_BASE", DEFAULT_BASE_URL).rstrip("/")

    @property
    def _model(self) -> str:
        return read_env("REPLICATE_MODEL").strip().strip("/")

    def _extra_input(self) -> dict[str, Any]:
        raw = read_env("REPLICATE_EXTRA_INPUT", "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise RuntimeError(f"REPLICATE_EXTRA_INPUT is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("REPLICATE_EXTRA_INPUT must be a JSON object.")
        return parsed

    def build_payload(self, job: Any) -> dict[str, Any]:
        model_input: dict[str, Any] = {"prompt": getattr(job, "prompt", "")}
        model_input.update(self._extra_input())
        return {"input": model_input}

    def submit(self, client: httpx.Client, job: Any) -> str:
        response = client.post(
            f"{self._base_url}/v1/models/{self._model}/predictions",
            json=self.build_payload(job),
        )
        response.raise_for_status()
        remote_id = (response.json() or {}).get("id")
        if not remote_id:
            raise RuntimeError(f"Replicate returned no prediction id: {response.text[:300]}")
        return str(remote_id)

    def poll(self, client: httpx.Client, remote_id: str) -> PollResult:
        response = client.get(f"{self._base_url}/v1/predictions/{remote_id}")
        response.raise_for_status()
        payload = response.json() or {}
        status = str(payload.get("status", "")).lower()

        if status == "succeeded":
            video = self._extract_output(payload.get("output"))
            if not video:
                return PollResult(
                    state=PollState.FAILED,
                    error=f"Replicate succeeded but the output was not a video: "
                    f"{str(payload.get('output'))[:300]}",
                    raw=payload,
                )
            return PollResult(state=PollState.SUCCEEDED, video_url=video, raw=payload)
        if status == "failed":
            return PollResult(
                state=PollState.FAILED,
                error=str(payload.get("error") or "prediction failed"),
                raw=payload,
            )
        if status == "canceled":
            return PollResult(state=PollState.CANCELLED, raw=payload)

        label = {"starting": "starting", "processing": "processing"}.get(status, "queued")
        return PollResult(
            state=PollState.PENDING if status == "starting" else PollState.RUNNING,
            message=f"Replicate is {label}…",
            raw=payload,
        )

    @staticmethod
    def _extract_output(output: Any) -> str | None:
        """Replicate returns a URL string, a list, or a keyed object."""
        if isinstance(output, str) and output.startswith("http"):
            return output
        if isinstance(output, list):
            for item in output:
                if isinstance(item, str) and item.startswith("http"):
                    return item
        if isinstance(output, dict):
            for value in output.values():
                if isinstance(value, str) and value.startswith("http"):
                    return value
                if isinstance(value, list):
                    nested = ReplicateAdapter._extract_output(value)
                    if nested:
                        return nested
        return None


register_provider(replace(SPEC, adapter_cls=ReplicateAdapter))

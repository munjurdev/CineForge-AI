"""
Kling AI — the value option: published per-second pricing and 5-10s clips.

    POST {base}/v1/videos/text2video | image2video -> data.task_id
    GET  {base}/v1/videos/{mode}/{task_id}         -> task_status + videos[0].url

Auth differs from every other provider here: Kling issues no bearer token.
Instead a JWT is signed from your Access Key / Secret Key pair (HS256, 30
minute expiry) and sent as ``Authorization: Bearer <jwt>``.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import httpx

from .base import CloudVideoAdapter, PollResult, PollState
from .registry import register_provider
from .spec import CAP_ASPECT, CAP_I2V, CAP_NEGATIVE, ProviderSpec, read_env

DEFAULT_BASE_URL = "https://api.klingai.com"
TOKEN_LIFETIME_SECONDS = 1800

SPEC = ProviderSpec(
    key="kling",
    label="Kling AI",
    kind="cloud",
    order=40,
    env_keys=("KLING_ACCESS_KEY", "KLING_SECRET_KEY"),
    optional_env=("KLING_API_BASE", "KLING_MODEL"),
    config_hint=(
        "Create an Access Key / Secret Key pair at https://app.klingai.com/ "
        "and set KLING_ACCESS_KEY and KLING_SECRET_KEY."
    ),
    capabilities=frozenset({CAP_I2V, CAP_ASPECT, CAP_NEGATIVE}),
    max_duration=10.0,
    default_model="kling-v1-6",
    preset={"aspect_ratio": "16:9", "duration_seconds": 5.0},
    website="https://klingai.com/",
    docs_url="https://kling.ai/document-api/guides/get-started/quick-start",
    pricing="resource packages, roughly 20 credits per second (std)",
    estimated_seconds=120.0,
)


def _sign_token(access_key: str, secret_key: str) -> str:
    """Build the HS256 JWT Kling expects (PyJWT if available, else stdlib)."""
    now = int(time.time())
    claims = {"iss": access_key, "nbf": now - 5, "iat": now, "exp": now + TOKEN_LIFETIME_SECONDS}
    try:
        import jwt  # PyJWT

        return jwt.encode(claims, secret_key, algorithm="HS256", headers={"kid": access_key})
    except ImportError:  # pragma: no cover - PyJWT is in requirements.txt
        import base64
        import hashlib
        import hmac
        import json

        def _b64(data: bytes) -> bytes:
            return base64.urlsafe_b64encode(data).rstrip(b"=")

        header = _b64(
            json.dumps({"alg": "HS256", "typ": "JWT", "kid": access_key}, separators=(",", ":")).encode()
        )
        payload = _b64(json.dumps(claims, separators=(",", ":")).encode())
        signing_input = header + b"." + payload
        signature = _b64(hmac.new(secret_key.encode(), signing_input, hashlib.sha256).digest())
        return (signing_input + b"." + signature).decode("ascii")


class KlingAdapter(CloudVideoAdapter):
    """Kling text/image-to-video tasks."""

    name = "kling"

    def headers(self) -> dict[str, str]:
        token = _sign_token(read_env("KLING_ACCESS_KEY"), read_env("KLING_SECRET_KEY"))
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @property
    def _base_url(self) -> str:
        return read_env("KLING_API_BASE", DEFAULT_BASE_URL).rstrip("/")

    @property
    def _mode(self) -> str:
        return "image2video"

    @staticmethod
    def _duration(job: Any) -> str:
        # Kling only accepts 5 or 10 seconds.
        return "10" if float(getattr(job, "duration_seconds", 5) or 5) > 7 else "5"

    @staticmethod
    def _ratio(job: Any) -> str:
        ratio = (getattr(job, "aspect_ratio", "") or "").strip()
        if ratio in {"16:9", "9:16", "1:1"}:
            return ratio
        width = int(getattr(job, "width", 0) or 0)
        height = int(getattr(job, "height", 0) or 0)
        if width and height:
            if height > width:
                return "9:16"
            if width == height:
                return "1:1"
        return "16:9"

    def build_payload(self, job: Any) -> dict[str, Any]:
        import base64

        payload: dict[str, Any] = {
            "model_name": read_env("KLING_MODEL", SPEC.default_model),
            "prompt": getattr(job, "prompt", ""),
            "negative_prompt": getattr(job, "negative_prompt", "") or "",
            "aspect_ratio": self._ratio(job),
            "duration": self._duration(job),
        }
        image = self.reference_image(job)
        if image:
            data, _mime = image
            # Kling accepts either a URL or base64-encoded image data.
            payload["image"] = base64.b64encode(data).decode("ascii")
        return payload

    def submit(self, client: httpx.Client, job: Any) -> str:
        mode = self._mode if getattr(job, "reference_image", None) else "text2video"
        response = client.post(
            f"{self._base_url}/v1/videos/{mode}",
            json=self.build_payload(job),
        )
        response.raise_for_status()
        body = response.json() or {}
        if body.get("code") not in (None, 0):
            raise RuntimeError(
                f"Kling error {body.get('code')}: {body.get('message') or body.get('data')}"
            )
        remote_id = ((body.get("data") or {}).get("task_id")) or body.get("task_id")
        if not remote_id:
            raise RuntimeError(f"Kling returned no task id: {response.text[:300]}")
        return f"{mode}:{remote_id}"

    def poll(self, client: httpx.Client, remote_id: str) -> PollResult:
        mode, _, task_id = remote_id.partition(":")
        response = client.get(f"{self._base_url}/v1/videos/{mode}/{task_id}")
        response.raise_for_status()
        body = response.json() or {}
        if body.get("code") not in (None, 0):
            return PollResult(
                state=PollState.FAILED,
                error=f"Kling error {body.get('code')}: {body.get('message')}",
                raw=body,
            )

        data = body.get("data") or {}
        status = str(data.get("task_status", "")).lower()
        if status in {"succeed", "succeeded", "success"}:
            videos = (data.get("task_result") or {}).get("videos") or []
            video = videos[0].get("url") if videos and isinstance(videos[0], dict) else None
            if not video:
                return PollResult(
                    state=PollState.FAILED,
                    error=f"Kling finished without a video: {str(data)[:300]}",
                )
            return PollResult(state=PollState.SUCCEEDED, video_url=str(video), raw=body)
        if status in {"failed", "fail"}:
            return PollResult(
                state=PollState.FAILED,
                error=str(data.get("task_status_msg") or "generation failed"),
                raw=body,
            )
        if status == "cancelled":
            return PollResult(state=PollState.CANCELLED, raw=body)

        label = {"submitted": "queued", "processing": "rendering"}.get(status, "queued")
        return PollResult(
            state=PollState.PENDING if status == "submitted" else PollState.RUNNING,
            message=f"Kling is {label}…",
            raw=body,
        )


register_provider(replace(SPEC, adapter_cls=KlingAdapter))

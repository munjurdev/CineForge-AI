"""
Shared adapter for hosted video-generation APIs.

Every cloud provider in CineForge follows the same shape:

    submit a job  ->  poll until it finishes  ->  download the MP4

so the polling loop, progress reporting, cancellation handling, error
formatting and file bookkeeping live here once. A provider module only
implements three small hooks — ``headers()``, ``submit()`` and ``poll()`` —
plus payload construction.

Providers that answer synchronously (one request in, video bytes back) set
``synchronous = True`` and implement ``run_sync()`` instead.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

import logging
import mimetypes
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from django.conf import settings

from ..adapters import AdapterError, BaseAdapter
from ..models import GenerationJob
from .spec import (
    CAPABILITY_LABELS,
    CAP_GUIDANCE,
    CAP_I2V,
    CAP_NEGATIVE,
    CAP_RESOLUTION,
    CAP_SEED,
    CAP_STEPS,
    ProviderSpec,
    read_env,
)

logger = logging.getLogger("cineforge.generator.providers")


class PollState(str, Enum):
    """Provider-neutral job states."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PollResult:
    """Outcome of a single status check."""

    state: PollState
    video_url: str | None = None
    message: str = ""
    error: str | None = None
    raw: dict[str, Any] | None = field(default=None, repr=False)


class ProviderError(AdapterError):
    """A provider rejected or failed a job."""


class CloudVideoAdapter(BaseAdapter):
    """Template-method adapter for a hosted generation API."""

    #: ``GenerationJob.engine`` value; must match the provider spec key.
    name: str = ""
    #: True when the API answers in a single synchronous request.
    synchronous: bool = False

    # ------------------------------------------------------------------ spec
    @property
    def spec(self) -> ProviderSpec:
        from .registry import get_provider

        spec = get_provider(self.name)
        if spec is None:  # pragma: no cover - registration bug
            raise AdapterError(
                f"No provider spec registered for {self.name!r}. "
                "Every adapter must be registered through register_provider()."
            )
        return spec

    @property
    def human_name(self) -> str:
        return self.spec.label

    @property
    def can_run_locally(self) -> bool:
        return self.spec.kind == "local"

    @property
    def hardware_notes(self) -> list[str]:
        spec = self.spec
        notes = [f"Hosted API — requires {spec.api_key_env or 'credentials'}."]
        if spec.pricing:
            notes.append(f"Pricing: {spec.pricing}")
        if spec.docs_url:
            notes.append(f"Docs: {spec.docs_url}")
        return notes

    # ------------------------------------------------------------- validation
    def validate(self, job: Any) -> tuple[bool, str | None, list[str]]:
        spec = self.spec
        warnings: list[str] = []

        if not spec.is_enabled:
            return False, f"{spec.label} is disabled in configuration.", []

        missing = spec.missing_env
        if missing:
            hint = spec.config_hint or f"Add {', '.join(missing)} to your .env file."
            return (
                False,
                f"{spec.label} is not configured ({', '.join(missing)} missing).",
                [hint],
            )

        # A job that needs a reference image cannot run on a text-only model.
        if getattr(job, "reference_image", None) and not spec.supports(CAP_I2V):
            return (
                False,
                f"{spec.label} does not support image-to-video.",
                ["Pick an engine that accepts a reference image."],
            )

        duration = float(getattr(job, "duration_seconds", 0) or 0)
        if spec.max_duration and duration > spec.max_duration:
            return (
                False,
                f"{spec.label} accepts clips up to {spec.max_duration:g}s, "
                f"this job asks for {duration:g}s.",
                [f"Lower the duration to {spec.max_duration:g}s or less."],
            )

        # Tuning knobs the provider ignores: warn instead of failing.
        for capability, attribute in (
            (CAP_SEED, "seed"),
            (CAP_STEPS, "steps"),
            (CAP_GUIDANCE, "guidance_scale"),
            (CAP_NEGATIVE, "negative_prompt"),
        ):
            if spec.supports(capability):
                continue
            value = getattr(job, attribute, None)
            if value in (None, "", 0):
                continue
            if attribute == "negative_prompt" and value == self._default_negative_prompt():
                # The service layer fills this in for every job, so it is not
                # a user request and should not be reported as unsupported.
                continue
            warnings.append(
                f"{CAPABILITY_LABELS[capability]} is not supported by "
                f"{spec.label} and will be ignored."
            )

        if not spec.supports(CAP_RESOLUTION) and getattr(job, "aspect_ratio", ""):
            warnings.append(
                f"{spec.label} chooses its own resolution; "
                "width/height and aspect ratio are ignored."
            )

        return True, None, warnings

    # ------------------------------------------------------------ HTTP client
    def headers(self) -> dict[str, str]:
        """Request headers every provider call carries (override as needed)."""
        spec = self.spec
        return {
            "Authorization": f"Bearer {read_env(spec.api_key_env)}",
            "Accept": "application/json",
        }

    def build_client(self) -> httpx.Client:
        """Build the HTTP client. Tests replace this with a mock transport."""
        timeout = float(read_env("VIDEO_PROVIDER_HTTP_TIMEOUT", "120"))
        return httpx.Client(timeout=timeout, headers=self.headers())

    # -------------------------------------------------------------- lifecycle
    def submit(self, client: httpx.Client, job: Any) -> str:
        """Queue the job remotely and return the provider's job id."""
        raise NotImplementedError

    def poll(self, client: httpx.Client, remote_id: str) -> PollResult:
        """Check one provider job."""
        raise NotImplementedError

    def run_sync(self, client: httpx.Client, job: Any) -> bytes | str:
        """For synchronous providers: return video bytes or a URL."""
        raise NotImplementedError

    def download_headers(self) -> dict[str, str]:
        """Extra headers for fetching the finished video (signed URLs need
        none; token-authenticated CDNs do)."""
        return {}

    # ---------------------------------------------------------------- running
    def generate(self, job: Any) -> str | Path | None:
        spec = self.spec
        client = self.build_client()
        try:
            if self.synchronous:
                outcome = self.run_sync(client, job)
                if isinstance(outcome, bytes):
                    return self._store_bytes(job, outcome)
                if not outcome:
                    return None
                downloaded = self._download(client, outcome, job)
            else:
                self._progress(job, 0.03, f"Submitting to {spec.label}…")
                remote_id = self.submit(client, job)
                logger.info("%s job %s queued as %s", spec.key, job.id, remote_id)
                video_url = self._await_completion(client, remote_id, job)
                if not video_url:
                    return None
                downloaded = self._download(client, video_url, job)
            if downloaded is None:
                return None
            return self._store_file(job, downloaded)
        except ProviderError as exc:
            logger.error("%s rejected job %s: %s", spec.key, job.id, exc)
            self._fail(job, str(exc), "PROVIDER_ERROR")
            return None
        except httpx.HTTPStatusError as exc:
            detail = self._describe_http_error(exc)
            logger.error("%s HTTP error for job %s: %s", spec.key, job.id, detail)
            self._fail(job, detail, "PROVIDER_HTTP_ERROR")
            return None
        except Exception as exc:  # noqa: BLE001 — a provider must never wedge a worker
            logger.exception("%s failed for job %s", spec.key, job.id)
            self._fail(job, f"{spec.label} failed: {exc}", "PROVIDER_ERROR")
            return None
        finally:
            try:
                client.close()
            except Exception:  # pragma: no cover
                pass

    def _await_completion(
        self, client: httpx.Client, remote_id: str, job: Any
    ) -> str | None:
        """Poll the provider until the clip exists, the job fails, or the
        deadline expires. Returns the download URL."""
        spec = self.spec
        timeout = float(
            read_env(
                "VIDEO_PROVIDER_TIMEOUT_SECONDS",
                str(getattr(settings, "GENERATION_JOB_TIMEOUT_SECONDS", 1800)),
            )
        )
        interval = float(read_env("VIDEO_PROVIDER_POLL_SECONDS", "3"))
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        last_message = ""

        while time.monotonic() < deadline:
            if self._is_cancelled(job):
                logger.info("%s job %s cancelled while polling", spec.key, job.id)
                return None

            result = self.poll(client, remote_id)

            if result.state is PollState.SUCCEEDED:
                if not result.video_url:
                    self._fail(
                        job,
                        f"{spec.label} reported success but returned no video.",
                        "PROVIDER_NO_OUTPUT",
                    )
                    return None
                self._progress(job, 0.95, "Downloading the finished video…")
                return result.video_url

            if result.state in (PollState.FAILED, PollState.CANCELLED):
                detail = result.error or result.message or "generation failed"
                self._fail(job, f"{spec.label}: {detail}", "PROVIDER_FAILED")
                return None

            elapsed = time.monotonic() - started
            message = result.message or f"{spec.label} is rendering…"
            if message != last_message:
                last_message = message
                # Providers rarely expose a percentage, so grow the bar with
                # elapsed time against the provider's typical turnaround.
                fraction = min(1.0, elapsed / max(spec.estimated_seconds, 1.0))
                self._progress(
                    job,
                    min(0.92, 0.05 + fraction * 0.85),
                    f"{message} ({elapsed:.0f}s elapsed)",
                )
            time.sleep(interval)

        self._fail(
            job,
            f"{spec.label} did not finish within {timeout:g}s.",
            "PROVIDER_TIMEOUT",
        )
        return None

    # ------------------------------------------------------------------ files
    def _download(
        self, client: httpx.Client, url: str, job: Any
    ) -> Path | None:
        """Stream a remote video into the job's temporary directory."""
        temp_base = getattr(settings, "MEDIA_TEMP_DIR", Path("media/temp"))
        dest_dir = Path(temp_base) / job.id.hex
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1] or "video.mp4"
        if "." not in name:
            name += ".mp4"
        dest = dest_dir / name

        headers = self.download_headers()
        with client.stream("GET", url, headers=headers, timeout=300.0) as response:
            if response.status_code >= 400:
                response.read()
                raise ProviderError(
                    f"Downloading the video failed "
                    f"(HTTP {response.status_code}): {response.text[:300]}"
                )
            with open(dest, "wb") as handle:
                for chunk in response.iter_bytes(64 * 1024):
                    handle.write(chunk)

        if dest.stat().st_size < 512:
            dest.unlink(missing_ok=True)
            self._fail(job, f"{self.spec.label} returned an empty video file.", "PROVIDER_NO_OUTPUT")
            return None
        return dest

    def _store_bytes(self, job: Any, payload: bytes) -> str | None:
        if len(payload) < 512:
            self._fail(job, f"{self.spec.label} returned an empty video.", "PROVIDER_NO_OUTPUT")
            return None
        temp_base = getattr(settings, "MEDIA_TEMP_DIR", Path("media/temp"))
        dest_dir = Path(temp_base) / job.id.hex
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "video.mp4"
        dest.write_bytes(payload)
        return self._store_file(job, dest)

    def _store_file(self, job: Any, source: Path) -> str | None:
        """Move a finished clip into ``media/outputs/<job id>/``."""
        import shutil

        output_base = getattr(settings, "MEDIA_OUTPUTS_DIR", Path("media/outputs"))
        dest_dir = Path(output_base) / job.id.hex
        dest_dir.mkdir(parents=True, exist_ok=True)
        final = dest_dir / source.name
        if final.exists():
            final.unlink()
        try:
            shutil.move(str(source), str(final))
        except OSError:
            shutil.copy2(str(source), str(final))
        return str(final)

    # ----------------------------------------------------------------- helpers
    def reference_image(self, job: Any) -> tuple[bytes, str] | None:
        """Return ``(raw bytes, mime type)`` for the job's reference image."""
        image = getattr(job, "reference_image", None)
        if not image:
            return None
        payload: bytes | None = None
        try:
            if hasattr(image, "open"):
                image.open("rb")
                payload = image.read()
                image.seek(0)
        except Exception:  # pragma: no cover - storage quirks
            payload = None
        if not payload:
            try:
                payload = Path(image.path).read_bytes()
            except Exception:  # pragma: no cover - remote storage
                return None
        if not payload:
            return None
        mime = mimetypes.guess_type(getattr(image, "name", "") or "")[0]
        return payload, mime or "image/png"

    @staticmethod
    def _default_negative_prompt() -> str:
        try:
            from ..services import DEFAULT_NEGATIVE_PROMPT

            return DEFAULT_NEGATIVE_PROMPT
        except Exception:  # pragma: no cover - import guard
            return ""

    @staticmethod
    def _is_cancelled(job: Any) -> bool:
        if getattr(job, "pk", None) is None:
            return False
        status = (
            GenerationJob.objects.filter(pk=job.pk)
            .values_list("status", flat=True)
            .first()
        )
        return status in {GenerationJob.Status.CANCELLED, GenerationJob.Status.FAILED}

    @staticmethod
    def _progress(job: Any, value: float, message: str) -> None:
        from ..services import job_set_progress

        job_set_progress(job, max(0.0, min(1.0, value)), message)

    @staticmethod
    def _fail(job: Any, message: str, error_type: str) -> None:
        from ..services import job_fail

        job_fail(job, message, error_type=error_type)

    @staticmethod
    def _describe_http_error(exc: httpx.HTTPStatusError) -> str:
        """Turn a non-2xx response into an actionable message."""
        status = exc.response.status_code
        body = ""
        try:
            body = exc.response.text[:300]
        except Exception:  # pragma: no cover
            pass
        hint = ""
        if status in (401, 403):
            hint = " Check the API key for this provider."
        elif status == 429:
            hint = " Rate limit or credit balance reached."
        return f"HTTP {status}{hint}: {body}".strip()

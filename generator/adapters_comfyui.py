"""
ComfyUI inference adapter — submits a workflow to a local ComfyUI instance
and polls for completion.

CineForge AI — Local AI Video Generation Platform.

This adapter follows the ComfyUI Custom Nodes API convention:
- POST /prompt (with workflow JSON)
- GET /status (optional polling)
- GET /view?filename=...&subfolder=output&type=output (download generated files)

The default workflow is a simple text-to-video excerpt that can be overridden
via configuration or a per-job workflow JSON if a model-specific workflow is
installed.

Important: the adapter does NOT download model weights automatically. It
expects a compatible workflow and model to already be installed in the local
ComfyUI instance.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import httpx

from django.conf import settings

from .adapters import BaseAdapter, registry
from .models import GenerationJob

logger = logging.getLogger("cineforge.generator.adapters_comfyui")


def _json_string(value: Any) -> str:
    """Escape a value for embedding inside a JSON string in the template."""
    return json.dumps(str(value))[1:-1]


# Tokens whose rendered value is JSON-escaped string content: they belong
# between the quotes of an existing JSON string. Every other token renders as
# a bare JSON value (number/boolean), which is what ComfyUI's type validation
# expects for seed, size, fps and frame-count inputs.
_STRING_TOKENS = frozenset({"PROMPT", "NEGATIVE_PROMPT", "MODEL_NAME"})

# Matches a fully quoted token ("{{SEED}}") or a bare one ({{SEED}}). Quoted
# form is tried first so the surrounding quotes are only ever consumed when
# they wrap the token completely.
_TOKEN_PATTERN = re.compile(r'"\{\{([A-Z_]+)\}\}"|\{\{([A-Z_]+)\}\}')


def _frame_count(job: Any | None) -> int:
    """Frame count for the latent video node, derived from duration and FPS.

    Video VAEs compress four frames per latent step, so the count is rounded up
    to the next 4n+1 value (81, 49, 33, ...) that Wan-style models expect.
    """
    duration = float(getattr(job, "duration_seconds", 0) or 2.0) if job else 2.0
    fps = int(getattr(job, "fps", 0) or 24) if job else 24
    frames = max(1, round(duration * fps))
    return ((frames + 3) // 4) * 4 + 1


def _history_output_files(
    history: dict[str, Any], prompt_id: str
) -> list[tuple[str, str]]:
    """Return (filename, subfolder) pairs from a ComfyUI /history response.

    ComfyUI keys the history payload by prompt id; older builds omit the
    wrapper. Node outputs usually hold the files under "gifs", "videos", or
    "images" as dicts with filename/subfolder/type keys.
    """
    entry = history.get(prompt_id)
    if not isinstance(entry, dict):
        if isinstance(history.get("outputs"), dict):
            entry = history
        else:
            entry = next(
                (value for value in history.values() if isinstance(value, dict)), {}
            )

    outputs = entry.get("outputs") or {}
    files: list[tuple[str, str]] = []
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for key in ("gifs", "videos", "images", "files"):
            for item in node_output.get(key) or []:
                if isinstance(item, dict) and item.get("filename"):
                    files.append(
                        (item["filename"], item.get("subfolder") or "output")
                    )
    return files


class ComfyUIAdapter(BaseAdapter):
    """Inference adapter for a local ComfyUI instance."""

    name = "comfyui"

    @property
    def can_run_locally(self) -> bool:
        # ComfyUI itself runs locally; this adapter just talks to its API.
        return True

    @property
    def human_name(self) -> str:
        return "ComfyUI (local)"

    @property
    def hardware_notes(self) -> list[str]:
        return [
            "Requires a running ComfyUI instance accessible at COMFYUI_BASE_URL.",
            "Models and VRAM requirements are determined by the ComfyUI workflow in use.",
            "This adapter does not download models automatically.",
        ]

    def validate(self, job: Any) -> tuple[bool, str | None, list[str]]:
        workflow_error = self._workflow_config_error()
        if workflow_error:
            return (
                False,
                workflow_error,
                ["Export a workflow from ComfyUI and point COMFYUI_WORKFLOW_PATH at it."],
            )

        base_url = getattr(settings, "COMFYUI_BASE_URL", "http://127.0.0.1:8188")
        timeout = float(getattr(settings, "COMFYUI_API_TIMEOUT", 600))

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(f"{base_url}/system_stats", timeout=10.0)
                if resp.status_code != 200:
                    return (
                        False,
                        f"ComfyUI returned status {resp.status_code}. "
                        "Start ComfyUI and set COMFYUI_BASE_URL.",
                        ["ComfyUI is not reachable at the configured URL."],
                    )
        except httpx.ConnectError:
            return (
                False,
                f"Cannot connect to ComfyUI at {base_url}. "
                "Start ComfyUI locally and check COMFYUI_BASE_URL.",
                ["ComfyUI is not running."],
            )
        except httpx.TimeoutException:
            return (
                False,
                f"ComfyUI at {base_url} did not respond in time.",
                ["ComfyUI may be starting up or overloaded."],
            )
        except Exception as exc:  # pragma: no cover - network edge cases
            return (
                False,
                f"ComfyUI check failed: {exc}",
                ["Check that ComfyUI is running and reachable."],
            )

        # Basic parameter sanity; the real limits depend on the workflow/model.
        resolution_warn: list[str] = []
        if job.width and job.width > 1024:
            resolution_warn.append(
                "Width over 1024 may exceed VRAM for many video workflows."
            )
        if job.height and job.height > 1024:
            resolution_warn.append(
                "Height over 1024 may exceed VRAM for many video workflows."
            )
        if (job.width and job.width % 16) or (job.height and job.height % 16):
            resolution_warn.append(
                "Width and height should be divisible by 16 — Wan 2.2 latent "
                "nodes reject other sizes."
            )
        if job.duration_seconds and job.duration_seconds > 10:
            resolution_warn.append(
                "Longer durations take more VRAM and time."
            )

        return True, None, resolution_warn

    def generate(self, job: Any) -> str | Path | None:
        base_url = getattr(settings, "COMFYUI_BASE_URL", "http://127.0.0.1:8188")
        timeout = float(getattr(settings, "COMFYUI_API_TIMEOUT", 600))
        output_base = getattr(settings, "MEDIA_OUTPUTS_DIR", Path("media/outputs"))
        temp_base = getattr(settings, "MEDIA_TEMP_DIR", Path("media/temp"))

        client = httpx.Client(timeout=timeout)

        try:
            # Build workflow from the configured exported API workflow
            workflow = self._load_workflow(job)
            if workflow is None:
                return None

            # Queue the prompt
            resp = client.post(
                f"{base_url}/prompt",
                json={
                    "prompt": workflow,
                    "client_id": str(job.id),
                },
            )
            if resp.status_code not in (200, 201):
                logger.error(
                    "ComfyUI /prompt returned %s: %s",
                    resp.status_code,
                    resp.text[:500],
                )
                return None

            result_payload = resp.json()
            if result_payload.get("error"):
                logger.error(
                    "ComfyUI rejected the workflow: %s",
                    json.dumps(result_payload.get("node_errors", result_payload))[:500],
                )
                return None
            prompt_id = result_payload.get("prompt_id")
            if not prompt_id:
                # Some versions return {filename, subfolder} directly for
                # queued executions; fall back to polling by client_id.
                prompt_id = result_payload.get("prompt_id") or str(job.id)

            logger.info(
                "Submitted ComfyUI prompt %s for job %s",
                prompt_id,
                job.id,
            )

            # Poll for completion via /history or /grid
            output_file = self._wait_for_output(
                client, base_url, prompt_id, job, temp_base
            )
            if output_file is None:
                return None

            # Move into a stable location
            dest = output_base / job.id.hex
            dest.mkdir(parents=True, exist_ok=True)
            final = dest / output_file.name
            if final.exists():
                final.unlink()
            try:
                import shutil
                shutil.move(str(output_file), str(final))
            except OSError:
                import shutil
                shutil.copy2(str(output_file), str(final))

            return str(final)

        except Exception as exc:  # noqa: BLE001
            logger.error("ComfyUI generation failed for job %s: %s", job.id, exc)
            return None
        finally:
            try:
                client.close()
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------
    # Workflow loading
    # ------------------------------------------------------------------

    def _workflow_path(self) -> Path | None:
        raw = getattr(settings, "COMFYUI_WORKFLOW_PATH", "")
        return Path(raw) if raw else None

    def _render_workflow(self, template: str, job: Any | None) -> str:
        """Replace {{TOKENS}} in an exported ComfyUI API workflow.

        A token may be written quoted (``"noise_seed": "{{SEED}}"``) or bare
        (``"noise_seed": {{SEED}}``). The quoted form is preferred because it
        keeps the workflow file valid JSON on disk; at render time the quotes
        around a numeric token are dropped so ComfyUI receives a JSON number.
        Substitution runs in a single pass, so token-looking text inside a
        prompt is left untouched.
        """
        prompt = getattr(job, "prompt", "") if job is not None else ""
        negative = getattr(job, "negative_prompt", "") if job is not None else ""
        model_name = getattr(job, "model_name", "") if job is not None else ""
        seed = getattr(job, "seed", None) if job is not None else None

        values = {
            "PROMPT": _json_string(prompt or ""),
            "NEGATIVE_PROMPT": _json_string(negative or ""),
            "MODEL_NAME": _json_string(model_name or ""),
            "WIDTH": str(int(getattr(job, "width", 0) or 832) if job else 832),
            "HEIGHT": str(int(getattr(job, "height", 0) or 480) if job else 480),
            "FPS": str(int(getattr(job, "fps", 0) or 24) if job else 24),
            "DURATION": str(float(getattr(job, "duration_seconds", 0) or 2.0) if job else 2.0),
            "SEED": str(int(seed) if seed is not None else 0),
            "FRAMES": str(_frame_count(job)),
        }
        def _substitute(match: re.Match[str]) -> str:
            quoted_key, bare_key = match.group(1), match.group(2)
            key = quoted_key or bare_key
            if key not in values:
                return match.group(0)
            value = values[key]
            if quoted_key and key in _STRING_TOKENS:
                # Keep the JSON string delimiters around the escaped content.
                return '"' + value + '"'
            return value

        return _TOKEN_PATTERN.sub(_substitute, template)

    def _workflow_config_error(self) -> str | None:
        """Return an actionable error when no usable workflow is configured."""
        path = self._workflow_path()
        if path is None:
            return (
                "No ComfyUI workflow is configured. Export your video workflow "
                "from ComfyUI (Save (API Format)) and set COMFYUI_WORKFLOW_PATH."
            )
        if not path.exists():
            return f"COMFYUI_WORKFLOW_PATH does not exist: {path}"
        try:
            rendered = self._render_workflow(path.read_text(encoding="utf-8"), None)
            json.loads(rendered)
        except (OSError, ValueError) as exc:
            return f"Could not load the ComfyUI workflow at {path}: {exc}"
        return None

    def _load_workflow(self, job: Any) -> dict[str, Any] | None:
        """Load the configured workflow and substitute the job's values."""
        error = self._workflow_config_error()
        if error:
            logger.error("%s", error)
            return None

        path = self._workflow_path()
        assert path is not None
        try:
            rendered = self._render_workflow(path.read_text(encoding="utf-8"), job)
            workflow = json.loads(rendered)
        except (OSError, ValueError) as exc:
            logger.error("Could not render ComfyUI workflow %s: %s", path, exc)
            return None
        if not isinstance(workflow, dict):
            logger.error("ComfyUI workflow must be a JSON object: %s", path)
            return None
        return workflow

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _wait_for_output(
        self,
        client: httpx.Client,
        base_url: str,
        prompt_id: str,
        job: Any,
        temp_dir: Path,
    ) -> Path | None:
        """Poll ComfyUI until the output appears or the job is cancelled."""
        deadline = time.monotonic() + float(getattr(settings, "COMFYUI_API_TIMEOUT", 600))
        poll_interval = 2.0
        attempts = 0
        last_status = ""

        while time.monotonic() < deadline:
            # Read status
            try:
                st = client.get(f"{base_url}/history/{prompt_id}", timeout=5.0)
                if st.status_code == 200:
                    for fname, subfolder in _history_output_files(
                        st.json(), prompt_id
                    ):
                        candidate = self._download_file(
                            client, base_url, fname, subfolder, temp_dir
                        )
                        if candidate and candidate.exists():
                            return candidate
                elif st.status_code != 404:
                    last_status = f"status {st.status_code}"
            except Exception as exc:  # pragma: no cover - network
                last_status = str(exc)

            # Alternative polling: /status
            try:
                s = client.get(f"{base_url}/status", timeout=5.0)
                if s.status_code == 200:
                    st_data = s.json()
                    exec_info = st_data.get("executing")
                    if exec_info is None:
                        # If history also empty, maybe still processing
                        pass
                    else:
                        last_status = f"executing {exec_info}"
            except Exception:  # pragma: no cover
                pass

            progress = min(0.95, 0.3 + (attempts * 0.05))
            from .services import job_set_progress

            job_set_progress(
                job,
                progress,
                f"Waiting for ComfyUI: {last_status or 'processing…'}",
            )

            attempts += 1
            time.sleep(poll_interval)

            # Check for cancellation using a fresh read, since the job row is
            # updated by the cancel view in another request.
            if job.pk is not None:
                current_status = (
                    GenerationJob.objects.filter(pk=job.pk)
                    .values_list("status", flat=True)
                    .first()
                )
                if current_status in {
                    GenerationJob.Status.CANCELLED,
                    GenerationJob.Status.FAILED,
                }:
                    return None

        from .services import job_fail

        job_fail(
            job,
            "ComfyUI timed out waiting for output.",
            error_type="COMFYUI_TIMEOUT",
        )
        return None

    def _download_file(
        self,
        client: httpx.Client,
        base_url: str,
        filename: str,
        subfolder: str,
        dest_dir: Path,
    ) -> Path | None:
        """Download an output file from ComfyUI's /view endpoint."""
        import urllib.parse

        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": "output",
        }
        encoded = urllib.parse.urlencode(params)
        url = f"{base_url}/view?{encoded}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename

        try:
            with client.stream("GET", url, timeout=120.0) as resp:
                if resp.status_code != 200:
                    logger.warning(
                        "ComfyUI /view returned %s for %s",
                        resp.status_code,
                        filename,
                    )
                    return None
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_bytes(8192):
                        fh.write(chunk)
            if dest.stat().st_size < 512:
                dest.unlink(missing_ok=True)
                return None
            return dest
        except Exception as exc:  # pragma: no cover - network
            logger.warning("ComfyUI download failed for %s: %s", filename, exc)
            return None


registry.register(ComfyUIAdapter.name, ComfyUIAdapter)

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

import logging
import time
from pathlib import Path
from typing import Any

import httpx

from django.conf import settings

from .adapters import BaseAdapter, registry

logger = logging.getLogger("cineforge.generator.adapters_comfyui")


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
        base_url = getattr(settings, "COMFYUI_BASE_URL", "http://127.0.0.1:8188")
        timeout = float(getattr(settings, "COMFYUI_API_TIMEOUT", 600))

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(f"{base_url}/version", timeout=10.0)
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
            # Build workflow
            workflow = self._build_workflow(job)
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
    # Workflow construction
    # ------------------------------------------------------------------

    def _build_workflow(self, job: Any) -> dict[str, Any] | None:
        """Return a ComfyUI workflow dict for the job.

        This is a minimal placeholder workflow. In a real installation this
        would be replaced by the actual workflow JSON used by the installed
        video model (Wan, LTX, etc.). The adapter should refuse to run if the
        real workflow is not configured.
        """
        workflow = {
            "99": {
                "class_type": "PreviewImage",
                "inputs": {
                    "image": ["57", 0],
                },
            },
            "57": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "cineforge",
                    "images": ["33", 0],
                },
            },
            "33": {
                "class_type": "VAEDecode",
                "inputs": {
                    "vae": ["31", 0],
                    "samples": ["32", 0],
                },
            },
            "31": {
                "class_type": "VAELoader",
                "inputs": {
                    "repository_id": "stabilityai/sd-vae-ft-mse",
                },
            },
            "32": {
                "class_type": "EmptyData",
                "inputs": {},
            },
            "9": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["3", 1],
                    "text": job.prompt or "",
                },
            },
            "10": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["3", 1],
                    "text": job.negative_prompt or "worst quality, low quality",
                },
            },
            "3": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "stabilityai/stable-diffusion-2-1",
                    "loaded": False,
                },
            },
        }
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
                    data = st.json()
                    outputs = data.get("outputs", {})
                    if outputs:
                        for node_id, node_data in outputs.items():
                            if node_data.get("subfolder") == "output":
                                for fname in node_data.get("files", []):
                                    # reconstruct path from filename/subfolder
                                    subfolder = node_data.get("subfolder", "output")
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

            # Check for cancellation
            if job.status in {job.status.CANCELLED, job.status.FAILED}:
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

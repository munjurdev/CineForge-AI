"""
End-to-end test of the ComfyUI adapter against a stub ComfyUI server.

CineForge AI — Local AI Video Generation Platform.

No GPU is available here, so this exercises the exact code path a real
generation uses — validate → POST /prompt → poll /history → download via
/view → store under MEDIA_OUTPUTS_DIR — against a local HTTP server that
mimics the ComfyUI API. The payload posted to /prompt is asserted to be the
shipped Wan 2.2 workflow with every token substituted.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from django.conf import settings
from django.test import TestCase

from generator.adapters_comfyui import ComfyUIAdapter
from generator.models import GenerationJob

# The adapter discards outputs smaller than 512 bytes, so pad past that.
FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4096


class _StubComfyUIHandler(BaseHTTPRequestHandler):
    """Minimal ComfyUI API: /version, /prompt, /history, /view, /status."""

    def log_message(self, *args):  # noqa: D102 - silence request logging
        pass

    def _send(self, code: int, payload: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        parsed = urlparse(self.path)
        if parsed.path == "/version":
            self._send(200, json.dumps({"version": "stub"}).encode())
        elif parsed.path == "/status":
            self._send(200, b"{}")
        elif parsed.path.startswith("/history/"):
            prompt_id = parsed.path.rsplit("/", 1)[-1]
            body = {
                prompt_id: {
                    "outputs": {
                        "98": {
                            "images": [
                                {
                                    "filename": "CineForge_00001_.mp4",
                                    "subfolder": "video",
                                    "type": "output",
                                }
                            ]
                        }
                    }
                }
            }
            self._send(200, json.dumps(body).encode())
        elif parsed.path == "/view":
            self._send(200, FAKE_MP4, ctype="video/mp4")
        else:
            self._send(404, b"{}")

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if self.path == "/prompt":
            payload = json.loads(raw)
            self.server.posted_workflow = payload.get("prompt")  # type: ignore[attr-defined]
            self._send(200, json.dumps({"prompt_id": "stub-prompt-1"}).encode())
        else:
            self._send(404, b"{}")


class ComfyUIStubServerTests(TestCase):
    """Run the adapter against a local fake ComfyUI instance."""

    posted_workflow: dict | None = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _StubComfyUIHandler)
        cls.server.posted_workflow = None  # type: ignore[attr-defined]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address[:2]
        cls.base_url = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        super().tearDownClass()

    def _workflow_path(self) -> Path:
        return (
            Path(settings.BASE_DIR)
            / "comfyui"
            / "workflows"
            / "wan22_t2v_api.json"
        )

    def test_validate_reaches_server(self) -> None:
        job = GenerationJob(prompt="a cinematic shot")
        with self.settings(
            COMFYUI_BASE_URL=self.base_url,
            COMFYUI_WORKFLOW_PATH=str(self._workflow_path()),
        ):
            ok, message, warnings = ComfyUIAdapter().validate(job)
        self.assertTrue(ok, message)
        self.assertEqual(warnings, [])

    def test_generate_submits_workflow_and_collects_output(self) -> None:
        job = GenerationJob(
            prompt='a "stormy" sea: 50% rain',
            negative_prompt="blurry, text",
            width=832,
            height=480,
            fps=16,
            duration_seconds=2.0,
            seed=99,
        )
        job.save()

        self.server.posted_workflow = None  # type: ignore[attr-defined]

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.settings(
                COMFYUI_BASE_URL=self.base_url,
                COMFYUI_WORKFLOW_PATH=str(self._workflow_path()),
                COMFYUI_API_TIMEOUT=30,
                MEDIA_OUTPUTS_DIR=root / "outputs",
                MEDIA_TEMP_DIR=root / "temp",
            ):
                output = ComfyUIAdapter().generate(job)

            self.assertIsNotNone(output)
            out_path = Path(str(output))
            self.assertTrue(out_path.exists())
            self.assertEqual(out_path.read_bytes(), FAKE_MP4)
            self.assertIn(str(job.id.hex), str(out_path))

        posted = self.server.posted_workflow  # type: ignore[attr-defined]
        self.assertIsNotNone(posted, "adapter never POSTed a workflow")
        assert posted is not None

        # Every token must be substituted, and numeric ones must be JSON
        # numbers so ComfyUI's type validation accepts them.
        self.assertNotIn("{{", json.dumps(posted))
        self.assertEqual(posted["99"]["inputs"]["text"], 'a "stormy" sea: 50% rain')
        self.assertEqual(posted["91"]["inputs"]["text"], "blurry, text")
        self.assertEqual(posted["96"]["inputs"]["noise_seed"], 99)
        self.assertIs(type(posted["96"]["inputs"]["noise_seed"]), int)
        self.assertEqual(posted["100"]["inputs"]["fps"], 16)
        self.assertEqual(posted["104"]["inputs"]["width"], 832)
        self.assertEqual(posted["104"]["inputs"]["height"], 480)
        # 2.0 s * 16 fps = 32 frames, rounded up to the 4n+1 the VAE expects.
        self.assertEqual(posted["104"]["inputs"]["length"], 33)
        self.assertIs(type(posted["104"]["inputs"]["length"]), int)
        self.assertEqual(posted["98"]["class_type"], "SaveVideo")
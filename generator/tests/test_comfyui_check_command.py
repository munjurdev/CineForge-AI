"""
Tests for the comfyui_check management command.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import httpx
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase


WORKFLOW = {
    "9": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "{{PROMPT}}"},
    }
}


def _client(status: int = 200, text: str = "0.3.40"):
    client = mock.MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    client.get.return_value = mock.Mock(status_code=status, text=text)
    return client


class ComfyUICheckCommandTests(TestCase):
    def _workflow_path(self, tmp: str) -> Path:
        path = Path(tmp) / "workflow.json"
        path.write_text(json.dumps(WORKFLOW), encoding="utf-8")
        return path

    def test_missing_workflow_fails(self):
        with self.settings(COMFYUI_WORKFLOW_PATH=""):
            with self.assertRaises(CommandError) as ctx:
                call_command("comfyui_check", stdout=StringIO())
        self.assertIn("COMFYUI_WORKFLOW_PATH", str(ctx.exception))

    def test_missing_workflow_file_fails(self):
        with self.settings(COMFYUI_WORKFLOW_PATH="/nope/workflow.json"):
            with self.assertRaises(CommandError) as ctx:
                call_command("comfyui_check", stdout=StringIO())
        self.assertIn("does not exist", str(ctx.exception))

    def test_unreachable_server_fails(self):
        with TemporaryDirectory() as tmp:
            path = self._workflow_path(tmp)
            with self.settings(COMFYUI_WORKFLOW_PATH=str(path)):
                with mock.patch(
                    "generator.management.commands.comfyui_check.httpx.Client",
                    side_effect=httpx.ConnectError("refused"),
                ):
                    with self.assertRaises(CommandError) as ctx:
                        call_command("comfyui_check", stdout=StringIO())
        self.assertIn("Cannot reach ComfyUI", str(ctx.exception))

    def test_bad_status_fails(self):
        with TemporaryDirectory() as tmp:
            path = self._workflow_path(tmp)
            with self.settings(COMFYUI_WORKFLOW_PATH=str(path)):
                with mock.patch(
                    "generator.management.commands.comfyui_check.httpx.Client",
                    return_value=_client(status=502),
                ):
                    with self.assertRaises(CommandError) as ctx:
                        call_command("comfyui_check", stdout=StringIO())
        self.assertIn("502", str(ctx.exception))

    def test_success_reports_workflow_and_version(self):
        out = StringIO()
        with TemporaryDirectory() as tmp:
            path = self._workflow_path(tmp)
            with self.settings(
                COMFYUI_WORKFLOW_PATH=str(path),
                COMFYUI_BASE_URL="https://pod-8188.proxy.runpod.net/",
            ):
                with mock.patch(
                    "generator.management.commands.comfyui_check.httpx.Client",
                    return_value=_client(status=200, text="0.3.40"),
                ):
                    call_command("comfyui_check", stdout=out)

        output = out.getvalue()
        self.assertIn("Workflow OK", output)
        self.assertIn("ComfyUI reachable", output)
        self.assertIn("0.3.40", output)

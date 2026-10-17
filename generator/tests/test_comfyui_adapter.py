"""
Tests for the ComfyUI adapter's workflow configuration.

CineForge AI — Local AI Video Generation Platform.

These tests do not require a running ComfyUI instance.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import TestCase

from generator.adapters_comfyui import ComfyUIAdapter, _history_output_files
from generator.models import GenerationJob


WORKFLOW = {
    "9": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "{{PROMPT}}"},
    },
    "10": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "{{NEGATIVE_PROMPT}}"},
    },
    "11": {
        "class_type": "EmptyLatentVideo",
        "inputs": {
            "width": "{{WIDTH}}",
            "height": "{{HEIGHT}}",
            "fps": "{{FPS}}",
            "duration": "{{DURATION}}",
            "seed": "{{SEED}}",
        },
    },
}


class WorkflowConfigTests(TestCase):
    def test_missing_config_returns_actionable_error(self):
        with self.settings(COMFYUI_WORKFLOW_PATH=""):
            ok, message, warnings = ComfyUIAdapter().validate(
                GenerationJob(prompt="test")
            )
        self.assertFalse(ok)
        self.assertIn("COMFYUI_WORKFLOW_PATH", message)
        self.assertTrue(warnings)

    def test_missing_file_returns_error(self):
        with self.settings(COMFYUI_WORKFLOW_PATH="/nonexistent/workflow.json"):
            ok, message, _ = ComfyUIAdapter().validate(GenerationJob(prompt="x"))
        self.assertFalse(ok)
        self.assertIn("does not exist", message)

    def test_invalid_json_returns_error(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text("{not json", encoding="utf-8")
            with self.settings(COMFYUI_WORKFLOW_PATH=str(path)):
                ok, message, _ = ComfyUIAdapter().validate(GenerationJob(prompt="x"))
        self.assertFalse(ok)
        self.assertIn("Could not load", message)

    def test_valid_workflow_passes_config_check(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow.json"
            path.write_text(json.dumps(WORKFLOW), encoding="utf-8")
            with self.settings(COMFYUI_WORKFLOW_PATH=str(path)):
                self.assertIsNone(ComfyUIAdapter()._workflow_config_error())


class WorkflowRenderTests(TestCase):
    def _write(self, tmp: str) -> Path:
        path = Path(tmp) / "workflow.json"
        path.write_text(json.dumps(WORKFLOW), encoding="utf-8")
        return path

    def test_prompt_and_numbers_are_substituted(self):
        job = GenerationJob(
            prompt='a "stormy" sea: 50% rain',
            negative_prompt="blurry, text",
            width=640,
            height=360,
            fps=16,
            duration_seconds=3.0,
            seed=1234,
        )
        with TemporaryDirectory() as tmp:
            path = self._write(tmp)
            with self.settings(COMFYUI_WORKFLOW_PATH=str(path)):
                workflow = ComfyUIAdapter()._load_workflow(job)

        self.assertIsNotNone(workflow)
        self.assertEqual(
            workflow["9"]["inputs"]["text"], 'a "stormy" sea: 50% rain'
        )
        self.assertEqual(workflow["10"]["inputs"]["text"], "blurry, text")
        inputs = workflow["11"]["inputs"]
        self.assertEqual(inputs["width"], "640")
        self.assertEqual(inputs["height"], "360")
        self.assertEqual(inputs["fps"], "16")
        self.assertEqual(inputs["duration"], "3.0")
        self.assertEqual(inputs["seed"], "1234")

    def test_generate_without_workflow_returns_none(self):
        with self.settings(COMFYUI_WORKFLOW_PATH=""):
            self.assertIsNone(
                ComfyUIAdapter()._load_workflow(GenerationJob(prompt="x"))
            )


class HistoryParsingTests(TestCase):
    def test_wrapped_history_payload(self):
        history = {
            "prompt-1": {
                "outputs": {
                    "9": {"gifs": [{"filename": "clip.mp4", "subfolder": "output"}]}
                }
            }
        }
        self.assertEqual(
            _history_output_files(history, "prompt-1"),
            [("clip.mp4", "output")],
        )

    def test_unwrapped_history_payload_and_video_key(self):
        history = {
            "outputs": {
                "12": {"videos": [{"filename": "scene.mp4"}]},
            }
        }
        self.assertEqual(
            _history_output_files(history, "missing-id"),
            [("scene.mp4", "output")],
        )

    def test_empty_history(self):
        self.assertEqual(_history_output_files({}, "prompt-1"), [])

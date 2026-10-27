"""
Tests for the ComfyUI adapter's workflow configuration.

CineForge AI — Local AI Video Generation Platform.

These tests do not require a running ComfyUI instance.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from django.test import TestCase

from generator.adapters_comfyui import (
    ComfyUIAdapter,
    _frame_count,
    _history_output_files,
)
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
        # Quoted numeric tokens must render as JSON numbers, not strings —
        # ComfyUI validates INT/FLOAT inputs strictly.
        self.assertEqual(inputs["width"], 640)
        self.assertEqual(inputs["height"], 360)
        self.assertEqual(inputs["fps"], 16)
        self.assertEqual(inputs["duration"], 3.0)
        self.assertEqual(inputs["seed"], 1234)
        self.assertIs(type(inputs["seed"]), int)

    def test_bare_numeric_tokens_still_work(self):
        """The older unquoted form keeps rendering to the same numbers."""
        bare = (
            '{"1": {"class_type": "Sampler", "inputs": '
            '{"seed": {{SEED}}, "steps": {{FRAMES}}}}}'
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bare.json"
            path.write_text(bare, encoding="utf-8")
            with self.settings(COMFYUI_WORKFLOW_PATH=str(path)):
                workflow = ComfyUIAdapter()._load_workflow(
                    GenerationJob(prompt="x", seed=55, fps=16, duration_seconds=3.0)
                )
        self.assertIsNotNone(workflow)
        assert workflow is not None
        self.assertEqual(workflow["1"]["inputs"]["seed"], 55)
        self.assertEqual(workflow["1"]["inputs"]["steps"], 49)

    def test_prompt_containing_token_text_is_not_resubstituted(self):
        job = GenerationJob(prompt="use {{SEED}} for the seed", seed=7)
        with TemporaryDirectory() as tmp:
            path = self._write(tmp)
            with self.settings(COMFYUI_WORKFLOW_PATH=str(path)):
                workflow = ComfyUIAdapter()._load_workflow(job)
        assert workflow is not None
        self.assertEqual(workflow["9"]["inputs"]["text"], "use {{SEED}} for the seed")
        self.assertEqual(workflow["11"]["inputs"]["seed"], 7)

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


class FrameCountTests(TestCase):
    def test_frames_round_up_to_4n_plus_1(self):
        # 3.0 s * 16 fps = 48 -> 49 frames
        self.assertEqual(_frame_count(GenerationJob(duration_seconds=3.0, fps=16)), 49)
        # 2.0 s * 16 fps = 32 -> 33 frames
        self.assertEqual(_frame_count(GenerationJob(duration_seconds=2.0, fps=16)), 33)
        # 0.5 s * 12 fps = 6 -> 9 frames
        self.assertEqual(_frame_count(GenerationJob(duration_seconds=0.5, fps=12)), 9)

    def test_frames_defaults_without_job(self):
        self.assertEqual(_frame_count(None), 49)  # 2.0 s * 24 fps


class ShippedWorkflowTests(TestCase):
    """The bundled Wan 2.2 workflow must render to a valid ComfyUI payload."""

    def _path(self) -> Path:
        from django.conf import settings

        return (
            Path(settings.BASE_DIR)
            / "comfyui"
            / "workflows"
            / "wan22_t2v_api.json"
        )

    def test_shipped_workflow_exists_and_renders(self):
        path = self._path()
        self.assertTrue(path.exists(), f"missing shipped workflow: {path}")

        # The file itself must stay parseable so editors, linters and jq can
        # read it — every token is written inside quotes.
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(raw["104"]["inputs"]["length"], "{{FRAMES}}")

        job = GenerationJob(
            prompt='a "stormy" sea: 50% rain',
            negative_prompt="blurry, text",
            width=832,
            height=480,
            fps=16,
            duration_seconds=3.0,
            seed=7,
        )
        with self.settings(COMFYUI_WORKFLOW_PATH=str(path)):
            workflow = ComfyUIAdapter()._load_workflow(job)

        self.assertIsNotNone(workflow)
        assert workflow is not None
        self.assertNotIn("{{", json.dumps(workflow))

        # Numeric tokens must arrive as JSON numbers, not strings.
        latent = workflow["104"]["inputs"]
        self.assertEqual(latent["length"], 49)
        self.assertIs(type(latent["length"]), int)
        self.assertEqual(workflow["96"]["inputs"]["noise_seed"], 7)
        self.assertIs(type(workflow["96"]["inputs"]["noise_seed"]), int)
        self.assertEqual(workflow["100"]["inputs"]["fps"], 16)
        self.assertIs(type(workflow["100"]["inputs"]["fps"]), int)

        # Prompt substitution stays JSON-safe and the video node is one the
        # adapter reads back from /history.
        self.assertEqual(workflow["99"]["inputs"]["text"], 'a "stormy" sea: 50% rain')
        self.assertEqual(workflow["91"]["inputs"]["text"], "blurry, text")
        self.assertEqual(workflow["98"]["class_type"], "SaveVideo")

    def test_configured_workflow_path_passes_check(self):
        with self.settings(COMFYUI_WORKFLOW_PATH=str(self._path())):
            self.assertIsNone(ComfyUIAdapter()._workflow_config_error())

    def test_shipped_workflow_uses_the_4_step_lora(self):
        raw = json.loads(self._path().read_text(encoding="utf-8"))

        high, low = raw["105"], raw["106"]
        self.assertEqual(high["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(low["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(
            high["inputs"]["lora_name"],
            "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
        )
        self.assertEqual(
            low["inputs"]["lora_name"],
            "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
        )
        self.assertEqual(high["inputs"]["strength_model"], 1.0)

        # Each expert samples 2 of the 4 total steps through its own LoRA'd
        # model: 0→2 on high noise, then 2→end on low noise.
        first, second = raw["96"], raw["95"]
        self.assertEqual(first["inputs"]["model"], ["105", 0])
        self.assertEqual(second["inputs"]["model"], ["106", 0])
        self.assertEqual(
            (
                first["inputs"]["steps"],
                first["inputs"]["cfg"],
                first["inputs"]["start_at_step"],
                first["inputs"]["end_at_step"],
            ),
            (4, 1.0, 0, 2),
        )
        self.assertEqual(
            (second["inputs"]["steps"], second["inputs"]["cfg"], second["inputs"]["start_at_step"]),
            (4, 1.0, 2),
        )

    def test_shipped_workflow_has_no_dangling_links(self):
        raw = json.loads(self._path().read_text(encoding="utf-8"))
        for node_id, node in raw.items():
            for name, value in node.get("inputs", {}).items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(
                        value[0],
                        raw,
                        f"node {node_id} input {name} links to missing node {value[0]}",
                    )


class ValidateWarningTests(TestCase):
    """validate() only reaches its resolution checks once ComfyUI answers."""

    @staticmethod
    def _reachable():
        client = mock.MagicMock()
        client.__enter__.return_value = client
        client.get.return_value.status_code = 200
        return mock.patch(
            "generator.adapters_comfyui.httpx.Client", return_value=client
        )

    def _validate(self, job):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow.json"
            path.write_text(json.dumps(WORKFLOW), encoding="utf-8")
            with self.settings(COMFYUI_WORKFLOW_PATH=str(path)):
                with self._reachable():
                    return ComfyUIAdapter().validate(job)

    def test_resolution_not_divisible_by_16_warns(self):
        ok, message, warnings = self._validate(
            GenerationJob(prompt="x", width=830, height=480)
        )
        self.assertTrue(ok, message)
        self.assertTrue(
            any("divisible by 16" in w for w in warnings), warnings
        )

    def test_clean_resolution_has_no_divisibility_warning(self):
        ok, message, warnings = self._validate(
            GenerationJob(prompt="x", width=832, height=480)
        )
        self.assertTrue(ok, message)
        self.assertFalse(any("divisible by 16" in w for w in warnings), warnings)


class FreeTierWorkflowTests(TestCase):
    """The Wan 2.2 5B workflow used by the free Colab/GPU route."""

    def _path(self) -> Path:
        from django.conf import settings

        return (
            Path(settings.BASE_DIR)
            / "comfyui"
            / "workflows"
            / "wan22_5b_t2v_api.json"
        )

    def test_raw_file_is_valid_json(self):
        raw = json.loads(self._path().read_text(encoding="utf-8"))
        self.assertEqual(raw["55"]["inputs"]["width"], "{{WIDTH}}")
        self.assertEqual(raw["3"]["inputs"]["seed"], "{{SEED}}")

    def test_renders_with_correct_types_and_no_dangling_links(self):
        job = GenerationJob(
            prompt='a "stormy" sea',
            negative_prompt="blurry",
            width=832,
            height=480,
            fps=24,
            duration_seconds=3.0,
            seed=7,
        )
        with self.settings(COMFYUI_WORKFLOW_PATH=str(self._path())):
            workflow = ComfyUIAdapter()._load_workflow(job)
            self.assertIsNone(ComfyUIAdapter()._workflow_config_error())

        self.assertIsNotNone(workflow)
        assert workflow is not None
        self.assertNotIn("{{", json.dumps(workflow))

        latent = workflow["55"]["inputs"]
        self.assertEqual(
            (latent["width"], latent["height"], latent["length"]),
            (832, 480, 73),  # 3.0 s * 24 fps = 72 -> 4n+1
        )
        self.assertTrue(all(isinstance(v, int) for v in (
            latent["width"], latent["height"], latent["length"]
        )))
        self.assertEqual(workflow["3"]["inputs"]["seed"], 7)
        self.assertEqual(workflow["57"]["inputs"]["fps"], 24)
        self.assertEqual(workflow["6"]["inputs"]["text"], 'a "stormy" sea')

        for node_id, node in workflow.items():
            for name, value in node.get("inputs", {}).items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], workflow, f"{node_id}.{name}")

    def test_matches_the_official_5b_template_choices(self):
        """Model/VAE/sampler must stay the ones ComfyUI's template ships."""
        raw = json.loads(self._path().read_text(encoding="utf-8"))
        self.assertEqual(raw["37"]["inputs"]["unet_name"], "wan2.2_ti2v_5B_fp16.safetensors")
        self.assertEqual(raw["39"]["inputs"]["vae_name"], "wan2.2_vae.safetensors")
        self.assertEqual(raw["38"]["inputs"]["type"], "wan")
        self.assertEqual(raw["55"]["class_type"], "Wan22ImageToVideoLatent")
        self.assertEqual(raw["3"]["inputs"]["sampler_name"], "uni_pc")
        self.assertEqual(raw["3"]["inputs"]["cfg"], 5.0)
        self.assertEqual(raw["58"]["class_type"], "SaveVideo")
        # Single model: no high/low-noise pair and no LoRA nodes.
        self.assertFalse(any(
            node.get("class_type") in {"LoraLoaderModelOnly", "KSamplerAdvanced"}
            for node in raw.values()
        ))

    def test_colab_notebook_exists_and_parses(self):
        from django.conf import settings

        notebook = (
            Path(settings.BASE_DIR) / "tools" / "colab" / "comfyui_free_gpu.ipynb"
        )
        self.assertTrue(notebook.exists(), f"missing notebook: {notebook}")
        data = json.loads(notebook.read_text(encoding="utf-8"))
        self.assertEqual(data["nbformat"], 4)
        code_cells = [c for c in data["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 5)
        for cell in data["cells"]:
            self.assertTrue(str(cell["source"]).strip())

        # Every code cell must be valid Python once IPython magics are masked —
        # this catches escape sequences that the builder turned into real
        # newlines inside string literals.
        for index, cell in enumerate(code_cells):
            lines = []
            for line in cell["source"].split("\n"):
                stripped = line.lstrip()
                if stripped.startswith(("!", "%")) and not stripped.startswith("#"):
                    line = line[: len(line) - len(stripped)] + "pass  # " + stripped
                lines.append(line)
            try:
                compile("\n".join(lines), f"cell{index}", "exec")
            except SyntaxError as exc:  # pragma: no cover - failure path
                self.fail(f"notebook code cell {index} is not valid Python: {exc}")

        # The smoke-test cell must embed the same workflow CineForge submits.
        self.assertIn("WORKFLOW =", code_cells[-1]["source"])
        self.assertIn("wan22_5b_t2v_api.json", "\n".join(
            str(c["source"]) for c in data["cells"]
        ))

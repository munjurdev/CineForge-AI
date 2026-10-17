"""
Tests for inference adapters.

CineForge AI — Local AI Video Generation Platform.

These tests exercise adapter validation and the test adapter without requiring
a GPU, ffmpeg, or a running ComfyUI instance.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from django.test import TestCase, override_settings

from generator.adapters import (
    AdapterError,
    BaseAdapter,
    registry,
)
from generator.adapters_demo import TestAdapter


class AdapterRegistryTests(TestCase):
    def setUp(self):
        registry._adapters.clear()

    def tearDown(self):
        registry._adapters.clear()

    def test_register_and_get(self):
        class Dummy(BaseAdapter):
            name = "dummy"

            def validate(self, job):
                return True, None, []

            def generate(self, job):
                return "/tmp/out.mp4"

        registry.register("dummy", Dummy)
        adapter = registry.get("dummy")
        self.assertIsInstance(adapter, Dummy)
        self.assertEqual(adapter.name, "dummy")

    def test_get_unknown_adapter_raises(self):
        with self.assertRaises(AdapterError):
            registry.get("nonexistent")

    def test_fallback_to_test_adapter(self):
        adapter = registry.get("test")
        self.assertIsInstance(adapter, TestAdapter)


class TestAdapterTests(TestCase):
    def test_validate_minimum_dimensions(self):
        from generator.models import GenerationJob

        job = GenerationJob(width=16, height=16, duration_seconds=1.0, fps=12)
        ok, msg, warnings = TestAdapter().validate(job)
        self.assertTrue(ok)

    def test_validate_too_small_dimensions(self):
        from generator.models import GenerationJob

        job = GenerationJob(width=10, height=10, duration_seconds=1.0, fps=12)
        ok, msg, warnings = TestAdapter().validate(job)
        self.assertFalse(ok)
        self.assertIn("16×16", msg)

    def test_can_run_locally(self):
        self.assertTrue(TestAdapter().can_run_locally)

    def test_hardware_notes(self):
        notes = TestAdapter().hardware_notes
        self.assertTrue(any("CPU" in n for n in notes))


@override_settings(FFMPEG_BIN="ffmpeg")
class TestAdapterGenerationTests(TestCase):
    def test_generate_fails_without_ffmpeg(self):
        from generator.models import GenerationJob

        job = GenerationJob(
            prompt="A test prompt.",
            width=64,
            height=64,
            fps=12,
            duration_seconds=1.0,
        )

        with mock.patch("shutil.which", return_value=None):
            result = TestAdapter().generate(job)
            self.assertIsNone(result)

    def test_ffmpeg_command_is_filtergraph_safe(self):
        from generator.models import GenerationJob

        job = GenerationJob(
            prompt="a beach at sunset: 50% clouds, warm light",
            width=64,
            height=64,
            fps=12,
            duration_seconds=1.0,
        )
        commands: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            return mock.Mock(returncode=1, stdout="", stderr="boom")

        with mock.patch("shutil.which", return_value="ffmpeg"):
            with mock.patch("subprocess.run", side_effect=fake_run):
                TestAdapter().generate(job)

        command = next(cmd for cmd in commands if "-filter_complex" in cmd)
        graph = command[command.index("-filter_complex") + 1]
        # Unquoted expressions with commas break FFmpeg's filtergraph parser.
        self.assertNotIn("fontsize=min(", graph)
        self.assertIn("fontsize=10", graph)
        # drawtext has no "where" option.
        self.assertNotIn("where=", graph)
        self.assertIn("\\:", graph)
        self.assertIn("\\%", graph)

    def test_generate_returns_path_with_ffmpeg(self):
        from generator.models import GenerationJob

        job = GenerationJob(
            prompt="A test prompt.",
            width=64,
            height=64,
            fps=12,
            duration_seconds=1.0,
        )

        def fake_run(cmd, **kwargs):
            # The adapter writes to the last argument of the ffmpeg command.
            Path(cmd[-1]).write_bytes(b"0" * 4096)
            return mock.Mock(returncode=0, stdout="", stderr="")

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.settings(MEDIA_TEMP_DIR=tmp_path / "media_temp"):
                with mock.patch("shutil.which", return_value="ffmpeg"):
                    with mock.patch("subprocess.run", side_effect=fake_run):
                        result = TestAdapter().generate(job)

                self.assertIsNotNone(result)
                self.assertIsInstance(result, str)
                self.assertTrue(Path(result).exists())
                self.assertTrue(Path(result).is_relative_to(tmp_path))

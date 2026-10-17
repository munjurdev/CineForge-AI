"""
Tests for the generator service layer.

CineForge AI — Local AI Video Generation Platform.

These tests cover validation, normalization, and job lifecycle helpers that do
not require a GPU or a running Celery worker.
"""
from __future__ import annotations

from django.test import TestCase

from generator.models import GenerationJob
from generator.services import (
    validate_and_normalize_job,
    validate_and_normalize_job_from_form,
    error_is_user_actionable,
    job_set_progress,
    job_fail,
    job_succeed,
)


class JobValidationTests(TestCase):
    def test_empty_prompt_rejected(self):
        job = GenerationJob(prompt="")
        ok, msg, warnings = validate_and_normalize_job(job)
        self.assertFalse(ok)
        self.assertEqual(msg, "A prompt is required.")

    def test_whitespace_only_prompt_rejected(self):
        job = GenerationJob(prompt="   ")
        ok, msg, warnings = validate_and_normalize_job(job)
        self.assertFalse(ok)

    def test_valid_job_with_presets(self):
        job = GenerationJob(
            prompt="A test prompt.",
            engine=GenerationJob.Engine.TEST,
            model_name="",
            duration_seconds=None,
            width=None,
            height=None,
            fps=None,
            aspect_ratio="",
        )
        ok, msg, warnings = validate_and_normalize_job(job)
        self.assertTrue(ok)
        self.assertEqual(msg, None)
        self.assertEqual(warnings, [])
        self.assertEqual(job.model_name, "Test adapter (CPU-only demo)")
        self.assertEqual(job.duration_seconds, 0.5)
        self.assertEqual(job.width, 64)
        self.assertEqual(job.height, 64)
        self.assertEqual(job.fps, 12)
        self.assertEqual(job.aspect_ratio, "1:1")

    def test_width_and_height_normalization_from_preset(self):
        job = GenerationJob(prompt="test", engine=GenerationJob.Engine.COMFYUI)
        validate_and_normalize_job(job)
        self.assertEqual(job.width, 832)
        self.assertEqual(job.height, 480)
        self.assertEqual(job.fps, 24)
        self.assertEqual(job.duration_seconds, 2.0)
        self.assertEqual(job.aspect_ratio, "16:9")

    def test_width_too_small_rejected(self):
        job = GenerationJob(prompt="test", width=32)
        ok, msg, _ = validate_and_normalize_job(job)
        self.assertFalse(ok)
        self.assertEqual(msg, "Width must be at least 64 pixels.")

    def test_width_too_large_rejected(self):
        job = GenerationJob(prompt="test", width=2500)
        ok, msg, _ = validate_and_normalize_job(job)
        self.assertFalse(ok)
        self.assertEqual(msg, "Width must be at most 2048 pixels for local generation.")

    def test_duration_zero_rejected(self):
        job = GenerationJob(prompt="test", duration_seconds=0)
        ok, msg, _ = validate_and_normalize_job(job)
        self.assertFalse(ok)
        self.assertEqual(msg, "Duration must be greater than zero.")

    def test_fps_out_of_range_rejected(self):
        job = GenerationJob(prompt="test", fps=0)
        ok, msg, _ = validate_and_normalize_job(job)
        self.assertFalse(ok)
        self.assertEqual(msg, "FPS must be between 1 and 60.")

    def test_guidance_scale_out_of_range_rejected(self):
        job = GenerationJob(prompt="test", guidance_scale=25)
        ok, msg, _ = validate_and_normalize_job(job)
        self.assertFalse(ok)
        self.assertEqual(msg, "Guidance scale must be between 0 and 20.")

    def test_steps_out_of_range_rejected(self):
        job = GenerationJob(prompt="test", steps=600)
        ok, msg, _ = validate_and_normalize_job(job)
        self.assertFalse(ok)
        self.assertEqual(msg, "Steps must be between 1 and 500.")

    def test_warnings_for_long_duration(self):
        job = GenerationJob(prompt="test", duration_seconds=35)
        ok, msg, warnings = validate_and_normalize_job(job)
        self.assertTrue(ok)
        self.assertTrue(any("longer durations" in w.lower() for w in warnings))

    def test_warnings_for_non_multiple_of_eight(self):
        job = GenerationJob(prompt="test", width=100)
        ok, msg, warnings = validate_and_normalize_job(job)
        self.assertTrue(ok)
        self.assertTrue(any("multiple of 8" in w.lower() for w in warnings))


class JobLifecycleTests(TestCase):
    def test_job_set_progress(self):
        job = GenerationJob.objects.create(prompt="test", engine=GenerationJob.Engine.TEST)
        job_set_progress(job, 0.5, "Halfway")
        job.refresh_from_db()
        self.assertEqual(job.progress, 0.5)
        self.assertEqual(job.message, "Halfway")

    def test_job_set_progress_clamps(self):
        job = GenerationJob.objects.create(prompt="test", engine=GenerationJob.Engine.TEST)
        job_set_progress(job, 1.5, "Over")
        job.refresh_from_db()
        self.assertEqual(job.progress, 1.0)

        job_set_progress(job, -0.5, "Under")
        job.refresh_from_db()
        self.assertEqual(job.progress, 0.0)

    def test_job_fail(self):
        job = GenerationJob.objects.create(prompt="test", engine=GenerationJob.Engine.TEST)
        job_fail(job, "Something went wrong", error_type="TEST_ERROR")
        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.FAILED)
        self.assertEqual(job.message, "Something went wrong")
        self.assertEqual(job.error_type, "TEST_ERROR")

    def test_job_succeed_without_output_file(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "clip.mp4"
            src.write_bytes(b"fake")
            job = GenerationJob.objects.create(prompt="test", engine=GenerationJob.Engine.TEST)
            with self.settings(MEDIA_OUTPUTS_DIR=tmp_path / "outputs"):
                job_succeed(job, src)
            job.refresh_from_db()
            self.assertEqual(job.status, GenerationJob.Status.SUCCESS)
            self.assertEqual(job.progress, 1.0)
            self.assertIsNotNone(job.output_file)


class ErrorActionabilityTests(TestCase):
    def test_user_actionable_error_types(self):
        for error_type in ("INVALID_PROMPT", "INVALID_SETTINGS", "MISSING_IMAGE", "MISSING_MODEL", "MODEL_NOT_CONFIGURED", "RESOLUTION_OUT_OF_RANGE", "WARMUP_FAILED"):
            job = GenerationJob(status=GenerationJob.Status.FAILED, error_type=error_type, message="")
            self.assertTrue(error_is_user_actionable(job), error_type)

    def test_non_user_actionable_error_types(self):
        job = GenerationJob(status=GenerationJob.Status.FAILED, error_type="INTERNAL_ERROR", message="")
        self.assertFalse(error_is_user_actionable(job))

    def test_user_actionable_by_message_content(self):
        job = GenerationJob(status=GenerationJob.Status.FAILED, error_type="", message="Missing model configured.")
        self.assertTrue(error_is_user_actionable(job))

    def test_non_actionable_message(self):
        job = GenerationJob(status=GenerationJob.Status.FAILED, error_type="", message="TIMEOUT")
        self.assertFalse(error_is_user_actionable(job))


class FormsBridgeTests(TestCase):
    def test_validate_and_normalize_job_from_form(self):
        class FakeForm:
            cleaned_data = {
                "prompt": "A cinematic test.",
                "negative_prompt": "blurry",
                "engine": GenerationJob.Engine.TEST,
                "model_name": "Test",
                "duration_seconds": 1.0,
                "width": 64,
                "height": 64,
                "fps": 12,
                "aspect_ratio": "1:1",
                "seed": 1,
                "guidance_scale": 5.0,
                "steps": 1,
                "model_variant": "FP16",
                "reference_image": None,
            }

        job = validate_and_normalize_job_from_form(FakeForm())
        self.assertEqual(job.prompt, "A cinematic test.")
        self.assertEqual(job.engine, GenerationJob.Engine.TEST)
        self.assertEqual(job.model_name, "Test")
        self.assertEqual(job.model_variant, "FP16")
        self.assertTrue(job._warnings is None or isinstance(job._warnings, list))

    def test_validate_and_normalize_job_from_form_rejects_empty_prompt(self):
        class BadForm:
            cleaned_data = {"prompt": ""}

        with self.assertRaises(ValueError):
            validate_and_normalize_job_from_form(BadForm())

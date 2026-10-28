"""
Tests for Celery tasks.

CineForge AI — Local AI Video Generation Platform.

These tests verify task behavior without a running Celery worker by using the
CELERY_TASK_ALWAYS_EAGER setting.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from django.test import TestCase, override_settings

from cineforge.celery import debug_task, ping
from generator.adapters import AdapterError
from generator.models import GeneratedVideo, GenerationJob
from generator.tasks import report_job_progress, revoke_task, run_generation_task


@override_settings(CELERY_TASK_ALWAYS_EAGER=True, CELERY_TASK_EAGER_PROPAGATES=True)
class CeleryTaskTests(TestCase):
    def test_ping(self):
        result = ping.apply()
        self.assertEqual(result.result["status"], "ok")
        self.assertIn("worker", result.result)

    def test_debug_task(self):
        result = debug_task.apply()
        self.assertTrue(result.successful())

    def test_revoke_task(self):
        with mock.patch("celery.current_app.control.revoke") as revoke:
            revoke_task("task-id-123")
        revoke.assert_called_once_with(
            "task-id-123", terminate=True, signal="SIGTERM"
        )

    def test_run_generation_task_unknown_job(self):
        result = run_generation_task("00000000-0000-0000-0000-000000000000")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Unknown job")

    @mock.patch("generator.tasks.get_adapter")
    def test_run_generation_task_adapter_error(self, get_adapter):
        job = GenerationJob.objects.create(prompt="test", engine="UNKNOWN")
        get_adapter.side_effect = AdapterError("No adapter")

        result = run_generation_task(str(job.pk))

        self.assertEqual(result["status"], "error")
        self.assertIn("Adapter error", result["message"])
        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.FAILED)
        self.assertEqual(job.error_type, "ADAPTER_ERROR")

    @mock.patch("generator.tasks.get_adapter")
    def test_run_generation_task_success(self, get_adapter):
        job = GenerationJob.objects.create(
            prompt="test", engine=GenerationJob.Engine.TEST, model_name="Test adapter"
        )

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "out.mp4"
            source.write_bytes(b"fake video")

            adapter = mock.Mock()
            adapter.validate.return_value = (True, None, [])
            adapter.generate.return_value = str(source)
            get_adapter.return_value = adapter

            with self.settings(
                MEDIA_ROOT=tmp_path,
                MEDIA_OUTPUTS_DIR=tmp_path / "outputs",
                MEDIA_TEMP_DIR=tmp_path / "temp",
                # The voiceover step talks to edge-tts over the network; it has
                # its own tests, so keep this wiring test offline.
                VOICEOVER_ENABLED=False,
            ):
                result = run_generation_task(str(job.pk))

        self.assertEqual(result["status"], "success")
        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.SUCCESS)
        self.assertTrue(job.output_file)
        self.assertTrue(GeneratedVideo.objects.filter(job=job).exists())

    @mock.patch("generator.tasks.add_voiceover_to_video")
    @mock.patch("generator.tasks.get_adapter")
    def test_run_generation_task_adds_voiceover(self, get_adapter, add_voiceover):
        job = GenerationJob.objects.create(prompt="a cat walking", engine="TEST")

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "out.mp4"
            source.write_bytes(b"fake video")
            voiced = tmp_path / "out_voiced.mp4"
            voiced.write_bytes(b"fake voiced video")
            add_voiceover.return_value = voiced

            adapter = mock.Mock()
            adapter.validate.return_value = (True, None, [])
            adapter.generate.return_value = str(source)
            get_adapter.return_value = adapter

            with self.settings(
                MEDIA_ROOT=tmp_path,
                MEDIA_OUTPUTS_DIR=tmp_path / "outputs",
                MEDIA_TEMP_DIR=tmp_path / "temp",
                VOICEOVER_ENABLED=True,
            ):
                result = run_generation_task(str(job.pk))

        self.assertEqual(result["status"], "success")
        add_voiceover.assert_called_once()
        job.refresh_from_db()
        self.assertTrue(job.output_file.name.endswith("out_voiced.mp4"))

    def test_report_job_progress(self):
        job = GenerationJob.objects.create(prompt="test", engine="TEST")
        report_job_progress(str(job.pk), 0.5, "Halfway")
        job.refresh_from_db()
        self.assertEqual(job.progress, 0.5)
        self.assertEqual(job.message, "Halfway")

    def test_report_job_progress_unknown_job_is_noop(self):
        # Should not raise for jobs that disappeared between reads.
        report_job_progress("00000000-0000-0000-0000-000000000000", 0.5)

"""
Tests for the generator app models.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.test import TestCase

from generator.models import GenerationJob, GeneratedVideo


class GenerationJobModelTests(TestCase):
    def test_status_labels(self):
        job = GenerationJob(status=GenerationJob.Status.PENDING)
        self.assertEqual(job.get_status_display(), "Pending")

        job.status = GenerationJob.Status.RUNNING
        self.assertEqual(job.get_status_display(), "Running")

        job.status = GenerationJob.Status.SUCCESS
        self.assertEqual(job.get_status_display(), "Success")

        job.status = GenerationJob.Status.FAILED
        self.assertEqual(job.get_status_display(), "Failed")

        job.status = GenerationJob.Status.CANCELLED
        self.assertEqual(job.get_status_display(), "Cancelled")

        job.status = GenerationJob.Status.DEFERRED
        self.assertEqual(job.get_status_display(), "Deferred")

    def test_terminal_states(self):
        for status in (GenerationJob.Status.SUCCESS, GenerationJob.Status.FAILED, GenerationJob.Status.CANCELLED):
            job = GenerationJob(status=status)
            self.assertTrue(job.is_terminal, status)

        for status in (GenerationJob.Status.PENDING, GenerationJob.Status.RUNNING, GenerationJob.Status.DEFERRED):
            job = GenerationJob(status=status)
            self.assertFalse(job.is_terminal, status)

    def test_can_retry(self):
        for status in (GenerationJob.Status.FAILED, GenerationJob.Status.CANCELLED):
            job = GenerationJob(status=status)
            self.assertTrue(job.can_retry(), status)

        for status in (GenerationJob.Status.PENDING, GenerationJob.Status.RUNNING, GenerationJob.Status.SUCCESS, GenerationJob.Status.DEFERRED):
            job = GenerationJob(status=status)
            self.assertFalse(job.can_retry(), status)

    def test_can_be_cancelled(self):
        for status in (GenerationJob.Status.PENDING, GenerationJob.Status.RUNNING):
            job = GenerationJob(status=status)
            self.assertTrue(job.can_be_cancelled(), status)

        for status in (GenerationJob.Status.SUCCESS, GenerationJob.Status.FAILED, GenerationJob.Status.CANCELLED, GenerationJob.Status.DEFERRED):
            job = GenerationJob(status=status)
            self.assertFalse(job.can_be_cancelled(), status)

    def test_create_retry(self):
        original = GenerationJob.objects.create(
            prompt="A test prompt.",
            negative_prompt="blurry",
            engine=GenerationJob.Engine.TEST,
            model_name="Test adapter",
            duration_seconds=1.0,
            width=64,
            height=64,
            fps=12,
            seed=1,
            guidance_scale=5.0,
            steps=1,
            status=GenerationJob.Status.FAILED,
            message="Failed for testing.",
        )
        retry = original.create_retry()
        self.assertNotEqual(retry.pk, original.pk)
        self.assertEqual(retry.prompt, original.prompt)
        self.assertEqual(retry.engine, original.engine)
        self.assertEqual(retry.status, GenerationJob.Status.PENDING)
        self.assertIn("Retry", retry.message)


class GeneratedVideoModelTests(TestCase):
    def test_size_mb(self):
        video = GeneratedVideo(size_bytes=10 * 1024 * 1024)
        self.assertEqual(video.size_mb, 10.0)

    def test_duration_display(self):
        video = GeneratedVideo(duration_seconds=95.0)
        self.assertEqual(video.duration_display, "1:35")

    def test_natural_size(self):
        video = GeneratedVideo(size_bytes=1500)
        self.assertEqual(video.natural_size(), "1.5 KB")
        video.size_bytes = 1500 * 1024
        self.assertEqual(video.natural_size(), "1.5 MB")

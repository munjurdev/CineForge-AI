from __future__ import annotations

from unittest import mock

from django.test import TestCase
from django.urls import reverse

from generator.models import GenerationJob


class JobProgressViewTests(TestCase):
    @mock.patch("generator.views.queue_generation_job", return_value=(True, "started"))
    def test_text_to_video_redirects_to_processing_job(self, queue_job):
        response = self.client.post(
            reverse("generator:generate_video"),
            {"prompt": "A red fox walking through snow"},
        )

        self.assertEqual(response.status_code, 302)
        job = queue_job.call_args.args[0]
        self.assertEqual(
            response["Location"],
            reverse("generator:job_detail", kwargs={"pk": job.pk}),
        )
        self.assertTrue(queue_job.call_args.kwargs["background"])

    def test_job_status_reports_real_sampling_progress(self):
        job = GenerationJob.objects.create(
            prompt="A cinematic test",
            status=GenerationJob.Status.RUNNING,
            progress=0.4,
            message="Sampling step 8 of 20 (40%).",
        )

        response = self.client.get(
            reverse("generator:job_status", kwargs={"pk": job.pk})
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "RUNNING",
            "status_display": "Running",
            "message": "Sampling step 8 of 20 (40%).",
            "progress_percent": 40,
            "progress_known": True,
        })

    def test_job_detail_contains_live_processing_page(self):
        job = GenerationJob.objects.create(
            prompt="A cinematic test",
            status=GenerationJob.Status.RUNNING,
            message="Starting generation in the background…",
        )

        response = self.client.get(
            reverse("generator:job_detail", kwargs={"pk": job.pk})
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Creating your video")
        self.assertContains(response, "data-status-url")
        self.assertContains(response, "Waiting for sampler steps")

    @mock.patch("generator.views.queue_generation_job", return_value=(True, "started"))
    def test_retry_redirects_to_background_processing_job(self, queue_job):
        original = GenerationJob.objects.create(
            prompt="A cinematic test",
            status=GenerationJob.Status.CANCELLED,
        )

        response = self.client.post(
            reverse("generator:job_retry", kwargs={"pk": original.pk})
        )

        self.assertEqual(response.status_code, 302)
        retry = queue_job.call_args.args[0]
        self.assertEqual(
            response["Location"],
            reverse("generator:job_detail", kwargs={"pk": retry.pk}),
        )
        self.assertTrue(queue_job.call_args.kwargs["background"])

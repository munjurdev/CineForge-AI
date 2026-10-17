"""
Tests for the library app views.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.test import TestCase
from django.urls import reverse

from generator.models import GeneratedVideo


class LibraryViewTests(TestCase):
    def setUp(self):
        self.video = GeneratedVideo.objects.create(
            title="Test clip",
            prompt_summary="A test clip.",
            file="outputs/test.mp4",
            duration_seconds=3.0,
            width=640,
            height=360,
            size_bytes=2048,
        )

    def test_video_list(self):
        response = self.client.get(reverse("library:video_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test clip")

    def test_video_detail(self):
        response = self.client.get(
            reverse("library:video_detail", kwargs={"pk": self.video.pk})
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test clip")
        self.assertContains(response, "640 × 360")

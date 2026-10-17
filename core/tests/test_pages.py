"""
Smoke tests for every user-facing page.

CineForge AI — Local AI Video Generation Platform.

These tests render each page through the Django test client so template
syntax errors and missing context variables fail the suite.
"""
from __future__ import annotations

from django.test import TestCase
from django.urls import reverse

from films.models import FilmProject
from generator.models import GeneratedVideo, GenerationJob


class PageSmokeTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.job = GenerationJob.objects.create(
            prompt="A cinematic test prompt.",
            engine=GenerationJob.Engine.TEST,
            model_name="Test adapter",
        )
        cls.video = GeneratedVideo.objects.create(
            job=cls.job,
            title="Smoke test clip",
            prompt_summary="A test clip.",
            file="outputs/test.mp4",
            duration_seconds=2.0,
            width=64,
            height=64,
            size_bytes=2048,
        )
        cls.film = FilmProject.objects.create(title="Smoke test film")

    def assert_page_ok(self, url):
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200, url)
        self.assertContains(response, 'id="main-content"')
        return response

    def test_core_pages(self):
        self.assert_page_ok(reverse("core:home"))
        self.assert_page_ok(reverse("core:dashboard"))

    def test_generator_pages(self):
        self.assert_page_ok(reverse("generator:generate_video"))
        self.assert_page_ok(reverse("generator:image_to_video"))
        self.assert_page_ok(reverse("generator:job_list"))
        self.assert_page_ok(
            reverse("generator:job_detail", kwargs={"pk": self.job.pk})
        )

    def test_library_pages(self):
        self.assert_page_ok(reverse("library:video_list"))
        self.assert_page_ok(
            reverse("library:video_detail", kwargs={"pk": self.video.pk})
        )

    def test_film_pages(self):
        self.assert_page_ok(reverse("films:film_list"))
        self.assert_page_ok(reverse("films:film_new"))
        self.assert_page_ok(
            reverse("films:film_detail", kwargs={"pk": self.film.pk})
        )
        self.assert_page_ok(
            reverse("films:film_edit", kwargs={"pk": self.film.pk})
        )

    def test_system_pages(self):
        self.assert_page_ok(reverse("system_monitor:diagnostics"))
        self.assert_page_ok(reverse("system_monitor:model_setup"))
        self.assert_page_ok(reverse("system_monitor:settings"))

    def test_settings_page_shows_real_values(self):
        response = self.assert_page_ok(reverse("system_monitor:settings"))
        self.assertContains(response, "Default adapter")
        self.assertContains(response, "Media root")

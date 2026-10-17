"""
Tests for the films app views.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.test import TestCase, override_settings
from django.urls import reverse

from films.models import FilmProject, FilmScene


@override_settings(DEBUG=True)
class FilmViewsTests(TestCase):
    def setUp(self):
        self.project = FilmProject.objects.create(title="Test Film", description="A test.")

    def test_film_list(self):
        response = self.client.get(reverse("films:film_list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test Film")

    def test_film_new_get(self):
        response = self.client.get(reverse("films:film_new"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "New film project")

    def test_film_new_post(self):
        data = {"title": "New Film", "description": "A new film.", "project_note": "Notes."}
        response = self.client.post(reverse("films:film_new"), data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(FilmProject.objects.filter(title="New Film").exists())

    def test_film_detail(self):
        response = self.client.get(reverse("films:film_detail", kwargs={"pk": self.project.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test Film")

    def test_film_edit_get(self):
        response = self.client.get(reverse("films:film_edit", kwargs={"pk": self.project.pk}))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Edit film project")

    def test_film_edit_post(self):
        data = {"title": "Updated", "description": "Updated description.", "project_note": "Updated notes."}
        response = self.client.post(reverse("films:film_edit", kwargs={"pk": self.project.pk}), data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.project.refresh_from_db()
        self.assertEqual(self.project.title, "Updated")

    def test_scene_add(self):
        data = {
            "scene-title": "Scene 1",
            "scene-prompt": "Test prompt.",
            "scene-duration_seconds": 2.0,
            "scene-transition_in": "cut",
            "scene-transition_out": "cut",
        }
        response = self.client.post(reverse("films:scene_add", kwargs={"pk": self.project.pk}), data, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(FilmScene.objects.filter(project=self.project, title="Scene 1").exists())

    def test_scene_delete(self):
        scene = FilmScene.objects.create(project=self.project, order=0, title="To delete", prompt="x")
        response = self.client.post(reverse("films:scene_delete", kwargs={"project_pk": self.project.pk, "scene_pk": scene.pk}), follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(FilmScene.objects.filter(pk=scene.pk).exists())

"""
Tests for the films app models.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.test import TestCase

from films.models import CharacterReference, FilmProject, FilmScene


class FilmProjectModelTests(TestCase):
    def test_str(self):
        project = FilmProject(title="My Short Film")
        self.assertEqual(str(project), f"Film {project.id.hex[:8]} — My Short Film")

    def test_can_assemble_needs_two_scenes(self):
        project = FilmProject.objects.create(title="Test")
        self.assertFalse(project.can_assemble())

    def test_assembly_output_path(self):
        project = FilmProject()
        self.assertIsNone(project.assembly_output_path())


class FilmSceneModelTests(TestCase):
    def test_str(self):
        scene = FilmScene(title="Opening scene")
        self.assertEqual(str(scene), f"Scene {scene.id.hex[:8]} — Opening scene")

    def test_status_choices(self):
        scene = FilmScene(status=FilmScene.Status.SUCCESS)
        self.assertEqual(scene.get_status_display(), "Success")
        self.assertEqual(scene.status_label(), "Success")

    def test_status_label_pending(self):
        scene = FilmScene(status=FilmScene.Status.PENDING)
        self.assertEqual(scene.status_label(), "Pending")


class CharacterReferenceModelTests(TestCase):
    def test_str(self):
        char = CharacterReference(name="Ada")
        self.assertEqual(str(char), "Character — Ada")

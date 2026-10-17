"""
Tests for the film scene workflow: generation, sync, reorder, characters.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase, override_settings
from django.urls import reverse

from films.models import CharacterReference, FilmProject, FilmScene
from films.services import generate_scene_clip, move_scene, next_scene_order, sync_scene_from_job
from generator.models import GeneratedVideo, GenerationJob


class SceneOrderTests(TestCase):
    def setUp(self):
        self.project = FilmProject.objects.create(title="Ordering")

    def test_next_order_uses_max_not_count(self):
        FilmScene.objects.create(project=self.project, order=0, title="A", prompt="a")
        FilmScene.objects.create(project=self.project, order=2, title="B", prompt="b")
        self.assertEqual(next_scene_order(self.project), 3)

    def test_scene_add_does_not_collide_after_delete(self):
        FilmScene.objects.create(project=self.project, order=0, title="A", prompt="a")
        FilmScene.objects.create(project=self.project, order=1, title="B", prompt="b")
        FilmScene.objects.create(project=self.project, order=2, title="C", prompt="c")
        FilmScene.objects.get(title="B").delete()

        response = self.client.post(
            reverse("films:scene_add", kwargs={"pk": self.project.pk}),
            {
                "scene-title": "D",
                "scene-prompt": "d",
                "scene-duration_seconds": 2.0,
                "scene-transition_in": "cut",
                "scene-transition_out": "cut",
            },
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(FilmScene.objects.get(title="D").order, 3)

    def test_move_scene_up_down(self):
        first = FilmScene.objects.create(project=self.project, order=0, title="A", prompt="a")
        second = FilmScene.objects.create(project=self.project, order=1, title="B", prompt="b")

        self.assertTrue(move_scene(second, "up"))
        first.refresh_from_db()
        second.refresh_from_db()
        self.assertEqual((first.order, second.order), (1, 0))

        self.assertTrue(move_scene(second, "down"))
        first.refresh_from_db()
        second.refresh_from_db()
        self.assertEqual((first.order, second.order), (0, 1))

    def test_move_scene_respects_boundaries(self):
        scene = FilmScene.objects.create(project=self.project, order=0, title="A", prompt="a")
        self.assertFalse(move_scene(scene, "up"))
        self.assertFalse(move_scene(scene, "down"))
        self.assertFalse(move_scene(scene, "sideways"))

    def test_move_scene_view(self):
        first = FilmScene.objects.create(project=self.project, order=0, title="A", prompt="a")
        second = FilmScene.objects.create(project=self.project, order=1, title="B", prompt="b")
        response = self.client.post(
            reverse(
                "films:scene_move",
                kwargs={
                    "project_pk": self.project.pk,
                    "scene_pk": second.pk,
                    "direction": "up",
                },
            ),
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        first.refresh_from_db()
        self.assertEqual(first.order, 1)


@override_settings(INFERENCE_DEFAULT_ADAPTER="test")
class SceneGenerationTests(TestCase):
    def setUp(self):
        self.project = FilmProject.objects.create(title="Generation")
        self.character = CharacterReference.objects.create(
            name="Mira",
            description="short silver hair, red coat",
        )
        self.scene = FilmScene.objects.create(
            project=self.project,
            order=0,
            title="Opening",
            prompt="a lighthouse in a storm",
            character_reference=self.character,
        )

    @patch("generator.services.queue_generation_job", return_value=(True, "queued"))
    def test_generate_scene_clip_creates_linked_job(self, queue):
        queued, message = generate_scene_clip(self.scene)

        self.assertTrue(queued, message)
        self.scene.refresh_from_db()
        self.assertIsNotNone(self.scene.job_id)
        job = self.scene.job
        self.assertEqual(job.engine, GenerationJob.Engine.TEST)
        self.assertIn("a lighthouse in a storm", job.prompt)
        self.assertIn("Mira", job.prompt)
        self.assertEqual(self.scene.status, FilmScene.Status.PENDING)
        queue.assert_called_once()

    @patch("generator.services.queue_generation_job", return_value=(True, "queued"))
    def test_generate_scene_clip_refuses_while_running(self, queue):
        self.scene.job = GenerationJob.objects.create(
            prompt="x", status=GenerationJob.Status.RUNNING
        )
        self.scene.status = FilmScene.Status.RUNNING
        self.scene.save()

        queued, message = generate_scene_clip(self.scene)
        self.assertFalse(queued)
        self.assertIn("already being generated", message)
        queue.assert_not_called()

    @patch("films.views.generate_scene_clip", return_value=(True, "queued"))
    def test_scene_generate_view(self, generate):
        response = self.client.post(
            reverse(
                "films:scene_generate",
                kwargs={"pk": self.project.pk, "scene_pk": self.scene.pk},
            ),
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        generate.assert_called_once()

    def test_sync_success_copies_output_and_video(self):
        job = GenerationJob.objects.create(
            prompt="x",
            status=GenerationJob.Status.SUCCESS,
            output_file="outputs/abc/clip.mp4",
        )
        video = GeneratedVideo.objects.create(job=job, file="outputs/abc/clip.mp4")
        self.scene.job = job
        self.scene.save()

        updated = sync_scene_from_job(job, video=video)

        self.assertEqual(updated, 1)
        self.scene.refresh_from_db()
        self.assertEqual(self.scene.status, FilmScene.Status.SUCCESS)
        self.assertEqual(self.scene.source_video_id, video.pk)
        self.assertTrue(self.scene.output_file.name.endswith("clip.mp4"))

    def test_sync_failure_marks_scene_failed(self):
        job = GenerationJob.objects.create(
            prompt="x", status=GenerationJob.Status.FAILED
        )
        self.scene.job = job
        self.scene.save()

        sync_scene_from_job(job)

        self.scene.refresh_from_db()
        self.assertEqual(self.scene.status, FilmScene.Status.FAILED)

    def test_sync_ignores_unlinked_jobs(self):
        job = GenerationJob.objects.create(prompt="x")
        self.assertEqual(sync_scene_from_job(job), 0)

    @patch("generator.views.queue_generation_job", return_value=(True, "queued"))
    def test_retry_job_keeps_scene_link(self, queue):
        old_job = GenerationJob.objects.create(
            prompt="x",
            engine=GenerationJob.Engine.TEST,
            status=GenerationJob.Status.FAILED,
        )
        self.scene.job = old_job
        self.scene.status = FilmScene.Status.FAILED
        self.scene.save()

        response = self.client.post(
            reverse("generator:job_retry", kwargs={"pk": old_job.pk}), follow=True
        )

        self.assertEqual(response.status_code, 200)
        self.scene.refresh_from_db()
        self.assertNotEqual(self.scene.job_id, old_job.pk)
        self.assertEqual(self.scene.status, FilmScene.Status.PENDING)
        queue.assert_called_once()


class CharacterReferenceTests(TestCase):
    def test_list_and_create(self):
        response = self.client.get(reverse("films:character_list"))
        self.assertEqual(response.status_code, 200)

        response = self.client.post(
            reverse("films:character_list"),
            {"name": "Mira", "description": "silver hair", "notes": ""},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(CharacterReference.objects.filter(name="Mira").exists())

    def test_character_appears_on_film_detail(self):
        project = FilmProject.objects.create(title="Film")
        character = CharacterReference.objects.create(name="Mira", description="silver hair")
        FilmScene.objects.create(
            project=project,
            order=0,
            title="Scene",
            prompt="x",
            character_reference=character,
        )
        response = self.client.get(
            reverse("films:film_detail", kwargs={"pk": project.pk})
        )
        self.assertContains(response, "Mira")

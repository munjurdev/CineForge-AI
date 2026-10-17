"""
Tests for the films service layer.

CineForge AI — Local AI Video Generation Platform.

These tests cover film assembly validation without requiring FFmpeg or a GPU.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from django.test import TestCase, override_settings

from films.models import FilmProject, FilmScene
from films.services import assemble_film


class FilmAssemblyTests(TestCase):
    def test_assemble_fails_without_enough_scenes(self):
        project = FilmProject.objects.create(title="Test")
        result = assemble_film(project)
        self.assertFalse(result["success"])
        self.assertIn("two", result["message"].lower())

    def test_assemble_fails_with_one_scene(self):
        project = FilmProject.objects.create(title="Test")
        FilmScene.objects.create(project=project, order=0, status="SUCCESS")
        result = assemble_film(project)
        self.assertFalse(result["success"])

    @mock.patch("films.models.FilmProject.can_assemble", return_value=True)
    @mock.patch("films.services.shutil.which", return_value=None)
    def test_assemble_fails_without_ffmpeg(self, which, can_assemble):
        project = FilmProject.objects.create(title="Test")
        FilmScene.objects.create(
            project=project, order=0, status="SUCCESS", output_file="temp/a.mp4"
        )
        FilmScene.objects.create(
            project=project, order=1, status="SUCCESS", output_file="temp/b.mp4"
        )
        result = assemble_film(project)
        self.assertFalse(result["success"])
        self.assertIn("FFmpeg", result["message"])

    @override_settings(FFMPEG_BIN="ffmpeg")
    def test_assemble_fails_when_scene_clips_cannot_be_probed(self):
        project = FilmProject.objects.create(title="Test")
        for order in (0, 1):
            FilmScene.objects.create(
                project=project,
                order=order,
                status="SUCCESS",
                output_file=f"temp/scene{order}.mp4",
            )

        with mock.patch("generator.services.probe_video_file", return_value=None):
            result = assemble_film(project)

        self.assertFalse(result["success"])
        self.assertIn("scene", result["message"].lower())

    @override_settings(FFMPEG_BIN="ffmpeg")
    @mock.patch("films.services.shutil.which", return_value="ffmpeg")
    def test_assemble_success_happy_path(self, which):
        project = FilmProject.objects.create(title="Test")
        s1 = FilmScene.objects.create(project=project, order=0, status="SUCCESS")
        s2 = FilmScene.objects.create(project=project, order=1, status="SUCCESS")

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            media_root = tmp_path / "media"
            (media_root / "temp").mkdir(parents=True)
            p1 = media_root / "temp" / "scene1.mp4"
            p2 = media_root / "temp" / "scene2.mp4"
            p1.write_bytes(b"clip1")
            p2.write_bytes(b"clip2")
            s1.output_file = "temp/scene1.mp4"
            s2.output_file = "temp/scene2.mp4"
            s1.save()
            s2.save()

            with self.settings(
                MEDIA_ROOT=media_root,
                MEDIA_TEMP_DIR=media_root / "temp",
                MEDIA_OUTPUTS_DIR=media_root / "outputs",
            ):
                with mock.patch(
                    "generator.services.probe_video_file",
                    return_value={"duration": 1.0},
                ):
                    with mock.patch(
                        "films.services.subprocess.run",
                        return_value=mock.Mock(returncode=0, stdout="", stderr=""),
                    ):
                        with mock.patch.object(Path, "exists", return_value=True):
                            with mock.patch.object(
                                Path,
                                "stat",
                                return_value=mock.Mock(st_size=4096),
                            ):
                                with mock.patch.object(Path, "unlink", return_value=None):
                                    with mock.patch(
                                        "films.services.shutil.move",
                                        side_effect=lambda a, b: Path(b).write_bytes(
                                            b"final"
                                        ),
                                    ):
                                        result = assemble_film(project)

            self.assertTrue(result["success"])
            self.assertIn("successfully", result["message"].lower())

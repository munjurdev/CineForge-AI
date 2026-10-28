"""
Tests for FFmpeg crossfade transitions during film assembly.

CineForge AI — Local AI Video Generation Platform.

The unit tests mock FFmpeg so they always run. The integration test renders two
tiny clips and is skipped when ffmpeg/ffprobe are not installed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock, skipUnless

from django.conf import settings
from django.test import TestCase, override_settings

from films.models import FilmProject, FilmScene
from films.services import assemble_film, build_xfade_filtergraph

# Honour the configured binary too, so the bundled tools/bin/ffmpeg is used when
# FFmpeg is not installed system-wide.
FFMPEG = (
    os.environ.get("FFMPEG_BIN")
    or shutil.which("ffmpeg")
    or getattr(settings, "FFMPEG_BIN", None)
)
FFPROBE = (FFMPEG or "").replace("ffmpeg", "ffprobe")


class FilterGraphTests(TestCase):
    def _graph(self, durations, transitions, seconds=1.0):
        return build_xfade_filtergraph(
            durations,
            transitions,
            width=832,
            height=480,
            fps=24,
            transition_seconds=seconds,
        )

    def test_all_cuts_returns_none(self):
        self.assertIsNone(self._graph([2.0, 3.0], ["cut"]))
        self.assertIsNone(self._graph([2.0, 3.0], [""]))

    def test_unknown_transition_falls_back_to_cut(self):
        self.assertIsNone(self._graph([2.0, 3.0], ["bogus"]))

    def test_unknown_durations_returns_none(self):
        self.assertIsNone(self._graph([0.0, 3.0], ["fade"]))
        self.assertIsNone(self._graph([2.0], ["fade"]))

    def test_single_xfade_uses_clamped_duration_and_offset(self):
        graph = self._graph([4.0, 4.0], ["fade"])
        self.assertIsNotNone(graph)
        self.assertIn("xfade=transition=fade:duration=1.000:offset=3.000", graph)
        self.assertIn("[outv]", graph)
        # Every input is normalised before the transition is applied.
        self.assertIn("scale=832:480", graph)
        self.assertIn("setsar=1", graph)
        self.assertIn("fps=24", graph)

    def test_transition_duration_is_clamped_to_half_the_shortest_clip(self):
        graph = self._graph([0.4, 0.4], ["dissolve"], seconds=2.0)
        self.assertIn("xfade=transition=dissolve:duration=0.200:offset=0.200", graph)

    def test_mixed_cut_and_xfade_offsets(self):
        graph = self._graph([2.0, 3.0, 4.0], ["cut", "wipeleft"])
        self.assertIsNotNone(graph)
        self.assertIn("concat=n=2:v=1:a=0[c1]", graph)
        # 2s + 3s joined = 5s, then a 1s crossfade starts at 4s.
        self.assertIn("xfade=transition=wipeleft:duration=1.000:offset=4.000", graph)
        self.assertIn("[x2]null[outv]", graph)

    def test_short_transitions_list_is_padded_with_cuts(self):
        graph = self._graph([2.0, 2.0, 2.0], ["fade"])
        self.assertIn("xfade=transition=fade", graph)
        self.assertIn("concat=n=2:v=1:a=0[c2]", graph)

    # -- audio chain ------------------------------------------------------

    def _audio_graph(self, durations, transitions, audio, seconds=1.0, force=False):
        return build_xfade_filtergraph(
            durations,
            transitions,
            width=832,
            height=480,
            fps=24,
            transition_seconds=seconds,
            audio_present=audio,
            force=force,
        )

    def test_crossfade_gets_a_matching_audio_crossfade(self):
        graph = self._audio_graph([4.0, 4.0], ["fade"], [True, True])
        self.assertIn("apad,atrim=0:4.000[a0]", graph)
        self.assertIn("[a0][a1]acrossfade=d=1.000:c1=tri:c2=tri[ax1]", graph)
        self.assertIn("[ax1]anull[outa]", graph)

    def test_audio_uses_concat_for_cuts(self):
        graph = self._audio_graph(
            [2.0, 3.0, 4.0], ["cut", "wipeleft"], [True, True, True]
        )
        self.assertIn("[a0][a1]concat=n=2:v=0:a=1[ac1]", graph)
        # 2s + 3s joined, then a 1s crossfade — same timing as the video chain.
        self.assertIn("xfade=transition=wipeleft", graph)
        self.assertIn("acrossfade=d=1.000", graph)
        self.assertIn("[outa]", graph)

    def test_silent_scene_gets_silence_of_the_right_length(self):
        graph = self._audio_graph([2.0, 3.0], ["fade"], [True, False])
        self.assertIn(
            "anullsrc=channel_layout=stereo:sample_rate=48000,atrim=0:3.000",
            graph,
        )
        self.assertIn("[a1]", graph)

    def test_silent_film_builds_no_audio_chain(self):
        graph = self._audio_graph([2.0, 2.0], ["fade"], [False, False])
        self.assertIn("xfade", graph)
        self.assertNotIn("[outa]", graph)
        self.assertNotIn("acrossfade", graph)

    def test_force_builds_a_graph_for_an_all_cut_film(self):
        self.assertIsNone(self._audio_graph([2.0, 2.0], ["cut"], [True, False]))
        graph = self._audio_graph([2.0, 2.0], ["cut"], [True, False], force=True)
        self.assertIsNotNone(graph)
        self.assertIn("concat=n=2:v=1:a=0[c1]", graph)
        self.assertIn("[outa]", graph)
        self.assertIn("anullsrc", graph)


@override_settings(FFMPEG_BIN="ffmpeg", FILM_TRANSITION_SECONDS=1.0)
class AssembleTransitionCommandTests(TestCase):
    def setUp(self):
        self.project = FilmProject.objects.create(title="Transitions")

    def _scenes(self, tmp_root: Path, *, transition: str = "fade"):
        clips = tmp_root / "clips"
        clips.mkdir(parents=True)
        scenes = []
        for index in range(2):
            clip = clips / f"scene{index}.mp4"
            clip.write_bytes(b"fake clip data")
            scene = FilmScene.objects.create(
                project=self.project,
                order=index,
                status="SUCCESS",
                output_file=f"clips/scene{index}.mp4",
                transition_out=transition,
            )
            scenes.append(scene)
        return scenes

    def _settings(self, root: Path):
        return self.settings(
            MEDIA_ROOT=root,
            MEDIA_TEMP_DIR=root / "temp",
            MEDIA_OUTPUTS_DIR=root / "outputs",
        )

    @mock.patch("films.services.shutil.which", return_value="ffmpeg")
    def test_assemble_uses_filter_complex_for_crossfade(self, which):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "media"
            root.mkdir()
            self._scenes(root)
            captured: dict[str, list[str]] = {}

            def fake_run(cmd, **kwargs):
                captured["cmd"] = cmd
                Path(cmd[-1]).write_bytes(b"x" * 4096)
                return mock.Mock(returncode=0, stdout="", stderr="")

            with self._settings(root):
                with mock.patch(
                    "generator.services.probe_video_file",
                    return_value={
                        "duration": 2.0,
                        "width": 160,
                        "height": 120,
                        "fps": 10,
                    },
                ), mock.patch(
                    "films.services.subprocess.run", side_effect=fake_run
                ):
                    result = assemble_film(self.project)

            self.assertTrue(result["success"], result["message"])
            self.assertIn("crossfade", result["message"].lower())
            cmd = captured["cmd"]
            self.assertIn("-filter_complex", cmd)
            graph = cmd[cmd.index("-filter_complex") + 1]
            self.assertIn("xfade=transition=fade", graph)
            # The test clips carry no audio, so the render stays silent.
            self.assertIn("-an", cmd)
            self.assertEqual(cmd[cmd.index("-map") + 1], "[outv]")
            self.assertEqual(cmd[cmd.index("-pix_fmt") + 1], "yuv420p")

    @mock.patch("films.services.shutil.which", return_value="ffmpeg")
    def test_assemble_keeps_scene_audio_across_a_crossfade(self, which):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "media"
            root.mkdir()
            self._scenes(root)
            captured: dict[str, list[str]] = {}

            def fake_run(cmd, **kwargs):
                captured["cmd"] = cmd
                Path(cmd[-1]).write_bytes(b"x" * 4096)
                return mock.Mock(returncode=0, stdout="", stderr="")

            probes = {
                "duration": 2.0,
                "width": 160,
                "height": 120,
                "fps": 10,
                "has_audio": True,
            }
            with self._settings(root):
                with mock.patch(
                    "generator.services.probe_video_file", return_value=probes
                ), mock.patch(
                    "films.services.subprocess.run", side_effect=fake_run
                ):
                    result = assemble_film(self.project)

            self.assertTrue(result["success"], result["message"])
            self.assertIn("audio kept", result["message"])
            cmd = captured["cmd"]
            graph = cmd[cmd.index("-filter_complex") + 1]
            self.assertIn("acrossfade", graph)
            self.assertIn("[outa]", cmd)
            self.assertNotIn("-an", cmd)
            self.assertEqual(cmd[cmd.index("-c:a") + 1], "aac")

    @mock.patch("films.services.shutil.which", return_value="ffmpeg")
    def test_mixed_audio_on_hard_cuts_uses_the_filter_graph(self, which):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "media"
            root.mkdir()
            self._scenes(root, transition="cut")
            captured: dict[str, list[str]] = {}

            def fake_run(cmd, **kwargs):
                captured["cmd"] = cmd
                Path(cmd[-1]).write_bytes(b"x" * 4096)
                return mock.Mock(returncode=0, stdout="", stderr="")

            def probe(path, *args, **kwargs):
                # Only the first scene has a voiceover track.
                return {
                    "duration": 2.0,
                    "width": 160,
                    "height": 120,
                    "fps": 10,
                    "has_audio": "scene0" in str(path),
                }

            with self._settings(root):
                with mock.patch(
                    "generator.services.probe_video_file", side_effect=probe
                ), mock.patch(
                    "films.services.subprocess.run", side_effect=fake_run
                ):
                    result = assemble_film(self.project)

            self.assertTrue(result["success"], result["message"])
            cmd = captured["cmd"]
            # Stream copy cannot join files with different stream layouts.
            self.assertIn("-filter_complex", cmd)
            self.assertIn("anullsrc", cmd[cmd.index("-filter_complex") + 1])
            self.assertIn("[outa]", cmd)

    @mock.patch("films.models.FilmProject.can_assemble", return_value=True)
    @mock.patch("films.services.shutil.which", return_value="ffmpeg")
    def test_assemble_falls_back_to_cut_when_durations_unknown(self, which, can):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "media"
            root.mkdir()
            self._scenes(root)
            captured: dict[str, list[str]] = {}

            def fake_run(cmd, **kwargs):
                captured["cmd"] = cmd
                Path(cmd[-1]).write_bytes(b"x" * 4096)
                return mock.Mock(returncode=0, stdout="", stderr="")

            with self._settings(root):
                with mock.patch(
                    "generator.services.probe_video_file", return_value=None
                ), mock.patch(
                    "films.services.subprocess.run", side_effect=fake_run
                ):
                    result = assemble_film(self.project)

            self.assertTrue(result["success"], result["message"])
            self.assertIn("skipped", result["message"].lower())
            self.assertNotIn("-filter_complex", captured["cmd"])
            self.assertIn("-c", captured["cmd"])

    @mock.patch("films.services.shutil.which", return_value="ffmpeg")
    def test_assemble_falls_back_when_ffmpeg_lacks_xfade(self, which):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "media"
            root.mkdir()
            self._scenes(root)
            calls: list[list[str]] = []

            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                if len(calls) == 1:
                    return mock.Mock(
                        returncode=1,
                        stdout="",
                        stderr="No such filter: 'xfade'",
                    )
                Path(cmd[-1]).write_bytes(b"x" * 4096)
                return mock.Mock(returncode=0, stdout="", stderr="")

            with self._settings(root):
                with mock.patch(
                    "generator.services.probe_video_file",
                    return_value={"duration": 2.0, "width": 160, "height": 120},
                ), mock.patch(
                    "films.services.subprocess.run", side_effect=fake_run
                ):
                    result = assemble_film(self.project)

            self.assertTrue(result["success"], result["message"])
            self.assertIn("hard cuts", result["message"].lower())
            self.assertIn("4.3", result["message"])
            self.assertEqual(len(calls), 2)
            self.assertNotIn("-filter_complex", calls[1])
            self.assertIn("-f", calls[1])

    @mock.patch("films.services.shutil.which", return_value="ffmpeg")
    def test_assemble_uses_copy_path_for_all_cuts(self, which):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "media"
            root.mkdir()
            self._scenes(root, transition="cut")
            captured: dict[str, list[str]] = {}

            def fake_run(cmd, **kwargs):
                captured["cmd"] = cmd
                Path(cmd[-1]).write_bytes(b"x" * 4096)
                return mock.Mock(returncode=0, stdout="", stderr="")

            with self._settings(root):
                with mock.patch(
                    "generator.services.probe_video_file",
                    return_value={"duration": 2.0, "width": 160, "height": 120},
                ), mock.patch(
                    "films.services.subprocess.run", side_effect=fake_run
                ):
                    result = assemble_film(self.project)

            self.assertTrue(result["success"], result["message"])
            self.assertNotIn("-filter_complex", captured["cmd"])
            self.assertIn("-c", captured["cmd"])


@skipUnless(
    bool(FFMPEG) and shutil.which(FFMPEG) and shutil.which(FFPROBE),
    "ffmpeg/ffprobe not installed",
)
class RealFFmpegTransitionTests(TestCase):
    def test_renders_crossfaded_movie(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "media"
            clips = root / "clips"
            clips.mkdir(parents=True)

            for index, colour in enumerate(("red", "blue")):
                subprocess.run(
                    [
                        FFMPEG,
                        "-y",
                        "-loglevel", "error",
                        "-f", "lavfi",
                        "-i", f"color=c={colour}:s=160x120:d=1:r=10",
                        "-pix_fmt", "yuv420p",
                        str(clips / f"scene{index}.mp4"),
                    ],
                    check=True,
                )

            project = FilmProject.objects.create(title="Real transition")
            for index in range(2):
                FilmScene.objects.create(
                    project=project,
                    order=index,
                    status="SUCCESS",
                    output_file=f"clips/scene{index}.mp4",
                    transition_out="fade",
                )

            with self.settings(
                MEDIA_ROOT=root,
                MEDIA_TEMP_DIR=root / "temp",
                MEDIA_OUTPUTS_DIR=root / "outputs",
                FFMPEG_BIN=FFMPEG,
                FILM_TRANSITION_SECONDS=0.4,
            ):
                result = assemble_film(project)
                self.assertTrue(result["success"], result["message"])

                from generator.services import probe_video_file

                meta = probe_video_file(Path(result["output"]))

            final = Path(result["output"])
            self.assertTrue(final.exists())
            self.assertGreater(final.stat().st_size, 1024)
            self.assertIsNotNone(meta)
            # 1s + 1s clips overlapping by 0.4s -> about 1.6s.
            self.assertAlmostEqual(meta["duration"], 1.6, delta=0.2)
            # The source clips are silent, so the film is silent too.
            self.assertFalse(meta["has_audio"])

    def test_keeps_voiceover_audio_through_a_crossfade(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "media"
            clips = root / "clips"
            clips.mkdir(parents=True)

            for index, colour in enumerate(("red", "blue")):
                subprocess.run(
                    [
                        FFMPEG,
                        "-y",
                        "-loglevel", "error",
                        "-f", "lavfi",
                        "-i", f"color=c={colour}:s=160x120:d=1:r=10",
                        "-f", "lavfi",
                        "-i", "sine=frequency=440:duration=1",
                        "-shortest",
                        "-c:v", "libx264",
                        "-c:a", "aac",
                        "-pix_fmt", "yuv420p",
                        str(clips / f"scene{index}.mp4"),
                    ],
                    check=True,
                )

            project = FilmProject.objects.create(title="Voiced transition")
            for index in range(2):
                FilmScene.objects.create(
                    project=project,
                    order=index,
                    status="SUCCESS",
                    output_file=f"clips/scene{index}.mp4",
                    transition_out="fade",
                )

            with self.settings(
                MEDIA_ROOT=root,
                MEDIA_TEMP_DIR=root / "temp",
                MEDIA_OUTPUTS_DIR=root / "outputs",
                FFMPEG_BIN=FFMPEG,
                FILM_TRANSITION_SECONDS=0.4,
            ):
                result = assemble_film(project)
                self.assertTrue(result["success"], result["message"])

                from generator.services import probe_video_file

                meta = probe_video_file(Path(result["output"]))

            self.assertIn("audio kept", result["message"])
            self.assertIsNotNone(meta)
            self.assertTrue(meta["has_audio"])
            self.assertAlmostEqual(meta["duration"], 1.6, delta=0.2)

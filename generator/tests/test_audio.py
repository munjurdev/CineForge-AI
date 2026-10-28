"""
Tests for the automatic AI voiceover pipeline.

CineForge AI — Local AI Video Generation Platform.

The speech synthesis itself is stubbed (the real edge-tts call needs network);
the FFmpeg muxing is exercised for real and skipped when FFmpeg is unavailable.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from django.conf import settings
from django.test import SimpleTestCase, override_settings

from generator.audio import (
    add_voiceover_to_video,
    mux_audio,
    narration_text_for,
    synthesize_speech,
)


def _ffmpeg() -> str | None:
    """Return the configured FFmpeg binary, or one from PATH."""
    configured = str(getattr(settings, "FFMPEG_BIN", "ffmpeg"))
    return shutil.which(configured) or shutil.which("ffmpeg")


def _ffprobe() -> str | None:
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        return None
    candidate = Path(ffmpeg).with_name(
        Path(ffmpeg).name.replace("ffmpeg", "ffprobe", 1)
    )
    return shutil.which(str(candidate)) or shutil.which("ffprobe")


def _stream_types(path: Path) -> list[str]:
    """Return the codec types present in a media file."""
    probe = _ffprobe()
    if not probe:
        return []
    proc = subprocess.run(
        [
            probe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return []
    data = json.loads(proc.stdout or "{}")
    return [s.get("codec_type", "") for s in data.get("streams", [])]


class NarrationTextTests(SimpleTestCase):
    def test_camera_directions_are_removed(self):
        self.assertEqual(
            narration_text_for("(camera pans left) a boat on the river"),
            "a boat on the river",
        )

    def test_trailing_style_words_are_trimmed(self):
        self.assertEqual(
            narration_text_for("a girl flying a kite, 4k, cinematic, hyperrealistic"),
            "a girl flying a kite",
        )

    def test_empty_prompt_returns_empty_string(self):
        self.assertEqual(narration_text_for(""), "")
        self.assertEqual(narration_text_for(None), "")

    def test_long_prompt_is_capped_at_a_word_boundary(self):
        text = narration_text_for("word " * 200)
        self.assertLessEqual(len(text), 241)
        self.assertTrue(text.endswith("…"))
        self.assertFalse(text.endswith(" …"))

    def test_max_chars_can_be_overridden(self):
        text = narration_text_for("one two three four five six seven", max_chars=12)
        self.assertLessEqual(len(text), 13)


class SpeechSynthesisTests(SimpleTestCase):
    def test_blank_text_is_not_synthesised(self):
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "voice.mp3"
            self.assertFalse(synthesize_speech("   ", dest))
            self.assertFalse(dest.exists())


class MuxTests(SimpleTestCase):
    def setUp(self):
        if not _ffmpeg():
            self.skipTest("ffmpeg is not installed")
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_clip(self, seconds: float = 1.0) -> Path:
        clip = self.dir / "clip.mp4"
        subprocess.run(
            [
                _ffmpeg(), "-y",
                "-f", "lavfi",
                "-i", f"testsrc=size=320x240:rate=12:duration={seconds}",
                "-pix_fmt", "yuv420p",
                str(clip),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=True,
        )
        return clip

    def _make_speech(self) -> Path:
        audio = self.dir / "speech.mp3"
        subprocess.run(
            [
                _ffmpeg(), "-y",
                "-f", "lavfi",
                "-i", "sine=frequency=440:duration=0.4",
                str(audio),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=True,
        )
        return audio

    @override_settings(FFMPEG_BIN="definitely-not-ffmpeg")
    def test_mux_returns_none_without_ffmpeg(self):
        missing = self.dir / "missing.mp4"
        self.assertIsNone(
            mux_audio(missing, missing, self.dir / "out.mp4")
        )

    def test_mux_adds_an_audio_stream_and_keeps_the_video(self):
        clip = self._make_clip(1.0)
        success = mux_audio(clip, self._make_speech(), self.dir / "muxed.mp4")

        self.assertIsNotNone(success)
        self.assertTrue(success.exists())
        types = _stream_types(success)
        self.assertIn("video", types)
        self.assertIn("audio", types)


class AddVoiceoverTests(SimpleTestCase):
    def setUp(self):
        if not _ffmpeg():
            self.skipTest("ffmpeg is not installed")
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.clip = self.dir / "clip.mp4"
        subprocess.run(
            [
                _ffmpeg(), "-y",
                "-f", "lavfi",
                "-i", "testsrc=size=320x240:rate=12:duration=1",
                "-pix_fmt", "yuv420p",
                str(self.clip),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=True,
        )

    def tearDown(self):
        self._tmp.cleanup()

    @override_settings(VOICEOVER_ENABLED=False)
    def test_disabled_returns_the_original_clip(self):
        job = mock.Mock(prompt="a boat on the river")
        self.assertEqual(add_voiceover_to_video(job, self.clip), self.clip)
        self.assertTrue(self.clip.exists())

    @mock.patch("generator.audio.synthesize_speech", return_value=False)
    def test_synthesis_failure_keeps_the_silent_clip(self, _synth):
        job = mock.Mock(prompt="a boat on the river")
        self.assertEqual(add_voiceover_to_video(job, self.clip), self.clip)
        self.assertTrue(self.clip.exists())

    @mock.patch("generator.audio.synthesize_speech")
    def test_voiceover_is_muxed_into_a_new_file(self, synth):
        speech = self.dir / "speech.mp3"
        subprocess.run(
            [
                _ffmpeg(), "-y",
                "-f", "lavfi",
                "-i", "sine=frequency=440:duration=0.4",
                str(speech),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=True,
        )

        def fake_synthesise(text, dest):
            shutil.copy(speech, dest)
            return True

        synth.side_effect = fake_synthesise
        job = mock.Mock(prompt="a boat on the river, 4k, cinematic")

        result = add_voiceover_to_video(job, self.clip)

        self.assertEqual(result.name, "clip_voiced.mp4")
        self.assertTrue(result.exists())
        self.assertIn("audio", _stream_types(result))
        self.assertFalse(self.clip.exists())
        synth.assert_called_once()
        spoken = synth.call_args.args[0]
        self.assertEqual(spoken, "a boat on the river")

    @override_settings(VOICEOVER_ENABLED=True)
    def test_nothing_speakable_in_prompt_leaves_the_clip_alone(self):
        job = mock.Mock(prompt="4k, cinematic")
        self.assertEqual(add_voiceover_to_video(job, self.clip), self.clip)

"""
Tests for the generator app forms.

CineForge AI — Local AI Video Generation Platform.

These tests exercise business logic and error cases that do not require a GPU.
"""
from __future__ import annotations

import base64

from django.test import SimpleTestCase

from generator.forms import GenerationJobForm, ImageToVideoForm


class GenerationJobFormTests(SimpleTestCase):
    def test_valid_form(self):
        data = {
            "prompt": "A cinematic drone shot over a calm lake at dusk.",
            "negative_prompt": "blurry, distorted",
            "engine": "TEST",
            "model_name": "Test adapter",
            "duration_seconds": 2.0,
            "width": 832,
            "height": 480,
            "fps": 24,
            "aspect_ratio": "16:9",
            "seed": 12345,
            "guidance_scale": 7.5,
            "steps": 25,
        }
        form = GenerationJobForm(data)
        self.assertTrue(form.is_valid(), form.errors)

    def test_empty_prompt_is_invalid(self):
        form = GenerationJobForm({"prompt": ""})
        self.assertFalse(form.is_valid())
        self.assertIn("prompt", form.errors)

    def test_width_too_small_is_invalid(self):
        form = GenerationJobForm({"prompt": "test", "width": 32})
        self.assertFalse(form.is_valid())
        self.assertIn("width", form.errors)

    def test_width_too_large_is_invalid(self):
        form = GenerationJobForm({"prompt": "test", "width": 3000})
        self.assertFalse(form.is_valid())
        self.assertIn("width", form.errors)

    def test_non_multiple_of_eight_produces_warning(self):
        form = GenerationJobForm({"prompt": "test", "width": 100, "height": 100})
        self.assertTrue(form.is_valid(), form.errors)
        self.assertTrue(any(k == "resolution_align" for k, _ in form.warnings))

    def test_duration_zero_is_invalid(self):
        form = GenerationJobForm({"prompt": "test", "duration_seconds": 0})
        self.assertFalse(form.is_valid())
        self.assertIn("duration_seconds", form.errors)

    def test_negative_prompt_optional(self):
        form = GenerationJobForm({"prompt": "test"})
        self.assertTrue(form.is_valid(), form.errors)

    def test_guidance_scale_out_of_range_is_invalid(self):
        form = GenerationJobForm({"prompt": "test", "guidance_scale": 25})
        self.assertFalse(form.is_valid())
        self.assertIn("guidance_scale", form.errors)

    def test_steps_out_of_range_is_invalid(self):
        form = GenerationJobForm({"prompt": "test", "steps": 600})
        self.assertFalse(form.is_valid())
        self.assertIn("steps", form.errors)


class ImageToVideoFormTests(SimpleTestCase):
    def test_valid_form(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        # A real 1×1 PNG so ImageField validation can inspect the file.
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        image = SimpleUploadedFile("ref.png", png_bytes, content_type="image/png")
        data = {
            "prompt": "Animate the image to the right.",
            "engine": "TEST",
            "model_name": "Test adapter",
            "duration_seconds": 1.0,
            "width": 512,
            "height": 512,
            "fps": 12,
            "seed": None,
            "guidance_scale": 5.0,
        }
        form = ImageToVideoForm(data, {"reference_image": image})
        self.assertTrue(form.is_valid(), form.errors)

    def test_missing_image_is_invalid(self):
        form = ImageToVideoForm({})
        self.assertFalse(form.is_valid())
        self.assertIn("reference_image", form.errors)

    def test_unsupported_image_extension_is_invalid(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        image = SimpleUploadedFile("ref.txt", b"txt", content_type="text/plain")
        form = ImageToVideoForm({"prompt": "test"}, {"reference_image": image})
        self.assertFalse(form.is_valid())
        self.assertIn("reference_image", form.errors)

    def test_image_too_large_is_invalid(self):
        from django.core.files.uploadedfile import SimpleUploadedFile

        large = b"x" * (51 * 1024 * 1024)
        image = SimpleUploadedFile("large.png", large, content_type="image/png")
        form = ImageToVideoForm({"prompt": "test"}, {"reference_image": image})
        self.assertFalse(form.is_valid())
        self.assertIn("reference_image", form.errors)

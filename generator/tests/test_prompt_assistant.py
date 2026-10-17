"""
Tests for the offline rule-based prompt assistant.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.test import TestCase
from django.urls import reverse

from generator.prompt_assistant import MAX_IDEA_LENGTH, build_cinematic_prompt


class PromptAssistantTests(TestCase):
    def test_expands_idea_without_model(self):
        prompt = build_cinematic_prompt("a lighthouse in a storm")
        self.assertIn("a lighthouse in a storm", prompt)
        self.assertIn("coherent motion", prompt)

    def test_output_is_deterministic(self):
        idea = "a wolf walking through snow"
        self.assertEqual(
            build_cinematic_prompt(idea),
            build_cinematic_prompt("  a wolf   walking through snow  "),
        )

    def test_blank_idea_is_rejected(self):
        with self.assertRaises(ValueError):
            build_cinematic_prompt("   ")

    def test_long_idea_is_truncated(self):
        prompt = build_cinematic_prompt("x" * (MAX_IDEA_LENGTH + 500))
        self.assertIn("Shortened from a longer idea.", prompt)

    def test_trailing_punctuation_is_not_duplicated(self):
        prompt = build_cinematic_prompt("A city street at night.")
        self.assertNotIn("night..", prompt)


class PromptAssistantViewTests(TestCase):
    def test_post_returns_rewritten_prompt(self):
        response = self.client.post(
            reverse("generator:enhance_prompt"),
            {"prompt": "a robot painting a sunset"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["method"], "rule-based")
        self.assertIn("a robot painting a sunset", payload["prompt"])

    def test_post_with_blank_prompt_returns_400(self):
        response = self.client.post(reverse("generator:enhance_prompt"), {"prompt": ""})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_get_is_not_allowed(self):
        response = self.client.get(reverse("generator:enhance_prompt"))
        self.assertEqual(response.status_code, 405)

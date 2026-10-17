"""
Tests for the core app views.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.test import TestCase
from django.urls import reverse


class CoreViewsTests(TestCase):
    def test_home(self):
        response = self.client.get(reverse("core:home"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "CineForge AI")

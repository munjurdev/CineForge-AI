"""
Tests for the system_monitor service layer.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from django.test import TestCase

from system_monitor.services import detect_system_info


class SystemMonitorServiceTests(TestCase):
    def test_detect_system_info_returns_dict(self):
        info = detect_system_info()
        self.assertIsInstance(info, dict)
        self.assertIn("os", info)
        self.assertIn("tools", info)
        self.assertIn("model_suggestions", info)
        self.assertIn("warnings", info)
        self.assertIn("memory", info)
        self.assertIn("gpu", info)
        self.assertIn("disk", info)

    def test_model_suggestions_always_populated(self):
        info = detect_system_info()
        self.assertTrue(len(info["model_suggestions"]) >= 1)

    def test_tools_section_present(self):
        info = detect_system_info()
        self.assertIn("ffmpeg", info["tools"])
        self.assertIn("ffprobe", info["tools"])
        self.assertIn("comfyui_running", info["tools"])

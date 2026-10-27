"""
Validate the ComfyUI configuration before starting a real generation.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

import httpx
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from generator.adapters_comfyui import ComfyUIAdapter


class Command(BaseCommand):
    help = (
        "Check COMFYUI_BASE_URL and COMFYUI_WORKFLOW_PATH so real video "
        "generation fails fast with a clear message instead of a failed job."
    )

    def handle(self, *args, **options):
        adapter = ComfyUIAdapter()

        workflow_error = adapter._workflow_config_error()
        if workflow_error:
            raise CommandError(workflow_error)

        workflow_path = getattr(settings, "COMFYUI_WORKFLOW_PATH", "")
        self.stdout.write(self.style.SUCCESS(f"Workflow OK: {workflow_path}"))

        base_url = getattr(
            settings, "COMFYUI_BASE_URL", "http://127.0.0.1:8188"
        ).rstrip("/")
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{base_url}/system_stats")
        except httpx.HTTPError as exc:
            raise CommandError(
                f"Cannot reach ComfyUI at {base_url}: {exc}"
            ) from exc

        if response.status_code != 200:
            raise CommandError(
                f"ComfyUI at {base_url} returned HTTP {response.status_code}. "
                "Expose port 8188 and check the URL."
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"ComfyUI reachable at {base_url} (/system_stats returned HTTP 200)."
            )
        )
        self.stdout.write("Submit a short test prompt from the generation page.")

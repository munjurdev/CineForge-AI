from __future__ import annotations

import threading
from unittest import mock

import websocket
from django.test import TestCase

from generator.adapters_comfyui import (
    _comfyui_websocket_url,
    _listen_for_progress,
)
from generator.models import GenerationJob


class ComfyUIProgressTests(TestCase):
    def test_websocket_url_uses_tls_for_https_and_preserves_base_path(self):
        self.assertEqual(
            _comfyui_websocket_url(
                "https://gpu.example/proxy/?token=abc",
                "job-123",
            ),
            "wss://gpu.example/proxy/ws?token=abc&clientId=job-123",
        )

    def test_sampler_events_write_real_progress_to_job(self):
        job = GenerationJob.objects.create(
            prompt="A cinematic test",
            status=GenerationJob.Status.RUNNING,
        )
        stop = threading.Event()
        connected = threading.Event()
        connection = mock.MagicMock()
        receive_count = 0

        def receive():
            nonlocal receive_count
            receive_count += 1
            if receive_count == 1:
                return '{"type":"progress","data":{"value":5,"max":20,"prompt_id":"p1"}}'
            stop.set()
            raise websocket.WebSocketTimeoutException()

        connection.recv.side_effect = receive
        with mock.patch(
            "websocket.create_connection",
            return_value=connection,
        ):
            _listen_for_progress(
                "https://gpu.example",
                str(job.pk),
                job.pk,
                stop,
                connected,
            )

        self.assertTrue(connected.is_set())
        job.refresh_from_db()
        self.assertEqual(job.progress, 0.25)
        self.assertEqual(job.message, "Sampling step 5 of 20 (25%).")
        connection.close.assert_called_once()

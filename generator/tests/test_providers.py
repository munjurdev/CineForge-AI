"""
Tests for the cloud video-generation provider system.

These cover the registry (add/remove a backend without touching Django),
the shared capability validation, and the submit -> poll -> download loop
against a mock transport — no network, no API keys, no GPU.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

import base64
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings

from generator.models import GenerationJob
from generator.providers import (
    CloudVideoAdapter,
    ProviderError,
    ProviderSpec,
    RegistrationError,
    all_providers,
    engine_choices,
    get_provider,
    provider_keys,
    register_provider,
)
from generator.providers.huggingface import HuggingFaceAdapter
from generator.providers.kling import KlingAdapter, _sign_token
from generator.providers.luma import LumaAdapter
from generator.providers.replicate import ReplicateAdapter
from generator.providers.runway import RunwayAdapter
from generator.providers.veo import VeoAdapter

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwAEhQMCJ7p2AAAAAElFTkSuQmCC"
)


class RegistryTests(SimpleTestCase):
    def test_every_expected_backend_is_registered(self):
        self.assertTrue(
            {
                "veo",
                "runway",
                "luma",
                "kling",
                "replicate",
                "huggingface",
                "comfyui",
                "test",
            }.issubset(set(provider_keys()))
        )

    def test_cloud_providers_declare_credentials_and_docs(self):
        for spec in all_providers():
            if spec.kind != "cloud":
                continue
            self.assertTrue(spec.env_keys, f"{spec.key} declares no env keys")
            self.assertTrue(spec.website, f"{spec.key} declares no website")
            self.assertTrue(spec.docs_url, f"{spec.key} declares no docs url")
            self.assertTrue(spec.status_note())

    def test_engine_choices_lists_every_enabled_provider(self):
        choices = dict(engine_choices())
        for spec in all_providers():
            if spec.is_enabled:
                self.assertIn(spec.key, choices)
                self.assertIn(spec.label, choices[spec.key])

    def test_unconfigured_cloud_provider_names_the_missing_variable(self):
        with override_settings(REPLICATE_API_TOKEN="", REPLICATE_MODEL=""):
            choices = dict(engine_choices())
        self.assertIn("REPLICATE_API_TOKEN", choices["replicate"])

    def test_disabling_a_provider_removes_it_everywhere(self):
        with override_settings(VIDEO_PROVIDERS_DISABLED="luma, kling"):
            self.assertNotIn("luma", dict(engine_choices()))
            self.assertNotIn("kling", dict(engine_choices()))
            self.assertFalse(get_provider("luma").is_enabled)
            self.assertFalse(get_provider("kling").is_ready)

    def test_enabling_only_some_providers_hides_the_rest(self):
        with override_settings(VIDEO_PROVIDERS_ENABLED="veo"):
            self.assertEqual(set(dict(engine_choices())), {"veo"})

    def test_duplicate_registration_is_rejected(self):
        spec = ProviderSpec(key="comfyui", label="Duplicate", kind="local")
        with self.assertRaises(RegistrationError):
            register_provider(spec)

    def test_overlong_or_uppercase_keys_are_rejected(self):
        with self.assertRaises(RegistrationError):
            register_provider(ProviderSpec(key="a" * 21, label="Too long"))
        with self.assertRaises(RegistrationError):
            register_provider(ProviderSpec(key="VeO", label="Wrong case"))


class CapabilityValidationTests(TestCase):
    def _job(self, **kwargs) -> GenerationJob:
        defaults = {
            "prompt": "a red panda climbing a bamboo forest",
            "engine": "luma",
            "duration_seconds": 5.0,
        }
        defaults.update(kwargs)
        return GenerationJob.objects.create(**defaults)

    def test_missing_credentials_block_the_job_and_name_the_variable(self):
        with override_settings(LUMA_API_KEY=""):
            ok, message, warnings = LumaAdapter().validate(self._job())
        self.assertFalse(ok)
        self.assertIn("LUMA_API_KEY", message or "")
        self.assertTrue(warnings)

    def test_clip_longer_than_the_provider_limit_is_rejected(self):
        with override_settings(LUMA_API_KEY="key"):
            ok, message, _ = LumaAdapter().validate(
                self._job(duration_seconds=30.0)
            )
        self.assertFalse(ok)
        self.assertIn("9s", message or "")

    def test_reference_image_is_rejected_without_image_to_video(self):
        with override_settings(LUMA_API_KEY="key"):
            job = self._job()
            job.reference_image = SimpleUploadedFile("ref.png", PNG_1PX, "image/png")
            ok, message, _ = LumaAdapter().validate(job)
        self.assertFalse(ok)
        self.assertIn("image-to-video", message or "")

    def test_reference_image_is_accepted_by_an_i2v_provider(self):
        with override_settings(RUNWAY_API_KEY="key"):
            job = self._job(engine="runway")
            job.reference_image = SimpleUploadedFile("ref.png", PNG_1PX, "image/png")
            ok, message, warnings = RunwayAdapter().validate(job)
        self.assertTrue(ok, message)
        self.assertEqual(warnings, [])

    def test_unsupported_tuning_knobs_warn_but_do_not_block(self):
        with override_settings(REPLICATE_API_TOKEN="t", REPLICATE_MODEL="acme/model"):
            ok, message, warnings = ReplicateAdapter().validate(
                self._job(engine="replicate", steps=24, seed=7)
            )
        self.assertTrue(ok, message)
        joined = " ".join(warnings)
        self.assertIn("Sampler steps", joined)
        self.assertIn("Seed", joined)

    def test_supported_tuning_knobs_do_not_warn(self):
        with override_settings(HF_TOKEN="hf_x", HF_MODEL="Wan-AI/Wan2.2-TI2V-5B"):
            ok, message, warnings = HuggingFaceAdapter().validate(
                self._job(engine="huggingface", steps=24, seed=7, guidance_scale=5.0)
            )
        self.assertTrue(ok, message)
        self.assertEqual(warnings, [])


class _MockHTTPMixin:
    """Run an adapter against a canned HTTP backend."""

    def run_with(self, handler, job: GenerationJob, adapter=None):
        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport)
        adapter = adapter or ReplicateAdapter()
        with patch.object(
            CloudVideoAdapter, "build_client", lambda self: client
        ):
            return adapter, adapter.generate(job)


class ReplicateFlowTests(_MockHTTPMixin, TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name)
        self._settings = override_settings(
            REPLICATE_API_TOKEN="test-token",
            REPLICATE_MODEL="acme/wan-t2v",
            VIDEO_PROVIDER_POLL_SECONDS="0.01",
            MEDIA_OUTPUTS_DIR=root / "outputs",
            MEDIA_TEMP_DIR=root / "temp",
        )
        self._settings.enable()
        self.addCleanup(self._settings.disable)
        self.addCleanup(self._tmp.cleanup)

    @staticmethod
    def _handler(states):
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if request.method == "POST" and path.startswith("/v1/models/"):
                return httpx.Response(201, json={"id": "pred-1"})
            if path == "/v1/predictions/pred-1":
                state = states[0] if len(states) > 1 else states[0]
                if len(states) > 1:
                    states.pop(0)
                if state == "succeeded":
                    return httpx.Response(
                        200,
                        json={
                            "status": "succeeded",
                            "output": "https://cdn.example/video.mp4",
                        },
                    )
                if state == "failed":
                    return httpx.Response(
                        200, json={"status": "failed", "error": "out of credits"}
                    )
                return httpx.Response(200, json={"status": state})
            if path == "/video.mp4":
                return httpx.Response(
                    200,
                    content=b"\x00" * 4096,
                    headers={"content-type": "video/mp4"},
                )
            return httpx.Response(404, json={"detail": "not found"})

        return handler

    def test_submit_poll_download_saves_the_clip(self):
        job = GenerationJob.objects.create(
            prompt="a lighthouse in a storm", engine="replicate"
        )
        adapter, result = self.run_with(
            self._handler(["processing", "succeeded"]), job
        )
        self.assertIsNotNone(result, job.message)
        path = Path(result)
        self.assertTrue(path.exists())
        self.assertGreaterEqual(path.stat().st_size, 512)
        # Kept under media/outputs/<job id>/ like the local adapter does.
        self.assertEqual(path.parent.name, job.id.hex)
        job.refresh_from_db()
        self.assertNotEqual(job.status, GenerationJob.Status.FAILED)

    def test_provider_failure_marks_the_job_failed(self):
        job = GenerationJob.objects.create(
            prompt="a lighthouse in a storm", engine="replicate"
        )
        adapter, result = self.run_with(self._handler(["failed"]), job)
        self.assertIsNone(result)
        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.FAILED)
        self.assertEqual(job.error_type, "PROVIDER_FAILED")
        self.assertIn("out of credits", job.message)

    def test_http_error_is_reported_with_a_hint(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(
                    401, json={"detail": "invalid token"}
                )
            return httpx.Response(404)

        job = GenerationJob.objects.create(
            prompt="a lighthouse in a storm", engine="replicate"
        )
        adapter, result = self.run_with(handler, job)
        self.assertIsNone(result)
        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.FAILED)
        self.assertIn("401", job.message)
        self.assertIn("API key", job.message)


class SynchronousProviderTests(TestCase):
    """Hugging Face answers in a single request instead of polling."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name)
        self._settings = override_settings(
            HF_TOKEN="hf_test",
            HF_MODEL="Wan-AI/Wan2.2-TI2V-5B",
            HF_PROVIDER="hf-inference",
            MEDIA_OUTPUTS_DIR=root / "outputs",
            MEDIA_TEMP_DIR=root / "temp",
        )
        self._settings.enable()
        self.addCleanup(self._settings.disable)
        self.addCleanup(self._tmp.cleanup)

    def test_video_bytes_are_stored(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertIn("Bearer hf_test", request.headers.get("authorization", ""))
            return httpx.Response(
                200, content=b"\x01" * 4096, headers={"content-type": "video/mp4"}
            )

        job = GenerationJob.objects.create(
            prompt="a paper boat crossing a puddle", engine="huggingface"
        )
        client = httpx.Client(
            transport=httpx.MockTransport(handler),
            headers={"Authorization": "Bearer hf_test"},
        )
        with patch.object(CloudVideoAdapter, "build_client", lambda self: client):
            result = HuggingFaceAdapter().generate(job)

        self.assertIsNotNone(result, job.message)
        self.assertTrue(Path(result).exists())

    def test_json_error_response_fails_the_job(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                503,
                json={"error": "model is loading"},
                headers={"content-type": "application/json"},
            )

        job = GenerationJob.objects.create(
            prompt="a paper boat crossing a puddle", engine="huggingface"
        )
        client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch.object(CloudVideoAdapter, "build_client", lambda self: client):
            result = HuggingFaceAdapter().generate(job)

        self.assertIsNone(result)
        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.FAILED)
        self.assertIn("model is loading", job.message)


class ProviderPayloadTests(SimpleTestCase):
    """Each provider maps the job onto its own documented request shape."""

    def test_veo_payload_carries_prompt_ratio_and_duration(self):
        job = GenerationJob(
            prompt="a camel crossing dunes",
            aspect_ratio="9:16",
            duration_seconds=8.0,
        )
        payload = VeoAdapter().build_payload(job)
        self.assertEqual(payload["instances"][0]["prompt"], "a camel crossing dunes")
        self.assertEqual(payload["parameters"]["aspectRatio"], "9:16")
        self.assertEqual(payload["parameters"]["durationSeconds"], 8)

    def test_veo_operation_url_handles_both_operation_shapes(self):
        adapter = VeoAdapter()
        with override_settings(
            VEO_API_KEY="k", VEO_BASE_URL="https://example.test/v1beta", VEO_MODEL="m"
        ):
            self.assertEqual(
                adapter._operation_url("operations/abc"),
                "https://example.test/v1beta/models/m/operations/abc",
            )
            self.assertEqual(
                adapter._operation_url("models/m/operations/abc"),
                "https://example.test/v1beta/models/m/operations/abc",
            )

    def test_runway_maps_landscape_and_portrait_onto_known_ratios(self):
        landscape = GenerationJob(width=1280, height=720, aspect_ratio="")
        portrait = GenerationJob(width=720, height=1280, aspect_ratio="")
        square = GenerationJob(width=960, height=960, aspect_ratio="")
        adapter = RunwayAdapter()
        self.assertEqual(adapter._ratio(landscape), "1280:720")
        self.assertEqual(adapter._ratio(portrait), "720:1280")
        self.assertEqual(adapter._ratio(square), "960:960")

    def test_luma_rounds_duration_onto_the_supported_values(self):
        self.assertEqual(LumaAdapter._duration(GenerationJob(duration_seconds=5)), "5s")
        self.assertEqual(LumaAdapter._duration(GenerationJob(duration_seconds=9)), "9s")

    def test_kling_duration_is_five_or_ten(self):
        self.assertEqual(KlingAdapter._duration(GenerationJob(duration_seconds=5)), "5")
        self.assertEqual(KlingAdapter._duration(GenerationJob(duration_seconds=10)), "10")

    def test_kling_token_is_a_valid_signed_jwt(self):
        import jwt

        token = _sign_token("my-access-key", "my-secret")
        header = jwt.get_unverified_header(token)
        claims = jwt.decode(token, "my-secret", algorithms=["HS256"])
        self.assertEqual(header["kid"], "my-access-key")
        self.assertEqual(claims["iss"], "my-access-key")
        self.assertGreater(claims["exp"], claims["iat"])

    def test_hugging_face_payload_uses_the_documented_field_names(self):
        job = GenerationJob(
            prompt="a kite in a windstorm",
            steps=20,
            guidance_scale=5.0,
            seed=42,
            duration_seconds=5.0,
            fps=16,
            negative_prompt="",
        )
        payload = HuggingFaceAdapter().build_payload(job)
        self.assertEqual(payload["inputs"], "a kite in a windstorm")
        self.assertEqual(payload["parameters"]["num_inference_steps"], 20)
        self.assertEqual(payload["parameters"]["guidance_scale"], 5.0)
        self.assertEqual(payload["parameters"]["seed"], 42)
        self.assertEqual(payload["parameters"]["num_frames"], 80)


class ModelDisplayTests(TestCase):
    def test_engine_display_comes_from_the_registry(self):
        job = GenerationJob.objects.create(prompt="x", engine="replicate")
        self.assertEqual(job.get_engine_display(), "Replicate")

    def test_engine_display_falls_back_for_unknown_values(self):
        job = GenerationJob.objects.create(prompt="x", engine="mystery")
        self.assertEqual(job.get_engine_display(), "mystery")


class FilmsEngineNormalisationTests(SimpleTestCase):
    def test_local_engines_keep_their_legacy_uppercase_form(self):
        from films.services import _normalise_engine

        self.assertEqual(_normalise_engine("comfyui"), "COMFYUI")
        self.assertEqual(_normalise_engine("test"), "TEST")

    def test_cloud_engines_keep_their_registered_lowercase_key(self):
        from films.services import _normalise_engine

        self.assertEqual(_normalise_engine("veo"), "veo")

    def test_unknown_engine_falls_back_to_comfyui(self):
        from films.services import _normalise_engine

        self.assertEqual(_normalise_engine("nope"), "COMFYUI")


class ProviderErrorTests(SimpleTestCase):
    def test_provider_error_is_an_adapter_error(self):
        from generator.adapters import AdapterError

        self.assertTrue(issubclass(ProviderError, AdapterError))

    def test_unknown_engine_error_explains_unimplemented_backends(self):
        from generator.adapters import AdapterError, get_adapter

        with self.assertRaises(AdapterError) as ctx:
            get_adapter("wan")
        self.assertIn("ComfyUI", str(ctx.exception))

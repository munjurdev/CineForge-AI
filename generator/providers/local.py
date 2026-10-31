"""
Local / first-party backends.

These engines already existed before the provider system, so their adapters
live in ``generator.adapters_comfyui`` and ``generator.adapters_demo``. This
module only declares their metadata (label, credentials, capabilities) so the
dropdown, dashboard and admin treat them exactly like cloud providers.

``WAN`` and ``LTX`` are historical engine values with no adapter
implementation yet; they stay listed for backwards compatibility and explain
themselves instead of failing silently.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from dataclasses import replace

from .registry import register_provider
from .spec import (
    CAP_ASPECT,
    CAP_GUIDANCE,
    CAP_I2V,
    CAP_NEGATIVE,
    CAP_RESOLUTION,
    CAP_SEED,
    CAP_STEPS,
    ProviderSpec,
)

_LOCAL_CAPABILITIES = frozenset(
    {CAP_I2V, CAP_ASPECT, CAP_RESOLUTION, CAP_SEED, CAP_STEPS, CAP_GUIDANCE, CAP_NEGATIVE}
)

COMFYUI = ProviderSpec(
    key="comfyui",
    label="ComfyUI (local)",
    kind="local",
    order=1,
    env_keys=(),
    optional_env=("COMFYUI_BASE_URL", "COMFYUI_WORKFLOW_PATH"),
    capabilities=_LOCAL_CAPABILITIES,
    default_model="ComfyUI workflow (select model in ComfyUI)",
    preset={
        "width": 832,
        "height": 480,
        "fps": 24,
        "duration_seconds": 5.0,
        "aspect_ratio": "16:9",
        "guidance_scale": 5.0,
        "steps": 28,
    },
    website="https://www.comfyui.org/",
    docs_url="https://docs.comfy.org/",
    pricing="free — runs on your own GPU (or the free Colab T4 workflow)",
    config_hint="Set COMFYUI_BASE_URL and COMFYUI_WORKFLOW_PATH.",
)

WAN = ProviderSpec(
    key="wan",
    label="Wan (direct)",
    kind="local",
    order=90,
    env_keys=(),
    capabilities=frozenset(),
    unavailable_note="No direct Wan adapter yet — use ComfyUI (local), which runs Wan 2.2.",
    config_hint="Use the ComfyUI engine to run Wan models.",
)

LTX = ProviderSpec(
    key="ltx",
    label="LTX-Video (direct)",
    kind="local",
    order=91,
    env_keys=(),
    capabilities=frozenset(),
    unavailable_note="No direct LTX-Video adapter yet — use ComfyUI (local).",
    config_hint="Use the ComfyUI engine to run LTX-Video models.",
)

TEST = ProviderSpec(
    key="test",
    label="Test adapter",
    kind="local",
    order=99,
    env_keys=(),
    capabilities=_LOCAL_CAPABILITIES,
    default_model="Test adapter (CPU-only demo)",
    preset={
        "width": 64,
        "height": 64,
        "fps": 12,
        "duration_seconds": 0.5,
        "aspect_ratio": "1:1",
        "guidance_scale": 5.0,
        "steps": 1,
    },
    pricing="free — CPU-only demo, renders a solid colour clip with FFmpeg",
)


def _register_locals() -> None:
    """Attach the real adapter classes (imported lazily to avoid a cycle)."""
    from ..adapters_comfyui import ComfyUIAdapter
    from ..adapters_demo import TestAdapter

    register_provider(replace(COMFYUI, adapter_cls=ComfyUIAdapter))
    register_provider(WAN)
    register_provider(LTX)
    register_provider(replace(TEST, adapter_cls=TestAdapter))


_register_locals()

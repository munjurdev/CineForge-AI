"""
Provider specifications — the metadata half of the plugin system.

A :class:`ProviderSpec` describes *one* video backend: where its credentials
come from, what it can do, roughly what it costs, and which adapter class
executes it. Everything that is not request/response plumbing lives here, so
adding a provider is a matter of writing one module that declares a spec.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Capability flags
# ---------------------------------------------------------------------------
# A provider declares the capabilities it actually supports. The shared
# adapter turns an unsupported capability into a validation error (for things
# a job cannot work without, like a reference image) or a warning (for tuning
# knobs the provider silently ignores, like sampler steps).

CAP_I2V = "i2v"  # accepts a reference image
CAP_AUDIO = "audio"  # generates its own soundtrack
CAP_SEED = "seed"
CAP_STEPS = "steps"
CAP_GUIDANCE = "guidance"
CAP_NEGATIVE = "negative"
CAP_ASPECT = "aspect_ratio"
CAP_RESOLUTION = "resolution"  # honours width/height from the job

# Human wording for the warnings the shared validator emits.
CAPABILITY_LABELS: dict[str, str] = {
    CAP_SEED: "Seed",
    CAP_STEPS: "Sampler steps",
    CAP_GUIDANCE: "Guidance scale",
    CAP_NEGATIVE: "Negative prompt",
}

#: Longest ``engine`` value the model column accepts (see GenerationJob).
MAX_ENGINE_KEY_LENGTH = 20


def read_env(name: str, default: str = "") -> str:
    """Read a configuration value from Django settings, the process
    environment, or the project ``.env`` file — in that order.

    Settings win so tests (and deployments) can override a provider with
    ``override_settings``; ``.env`` keeps secrets out of source code, which is
    how the rest of CineForge is configured.
    """
    try:  # pragma: no cover - import guard
        from django.conf import settings

        if settings.configured and hasattr(settings, name):
            value = getattr(settings, name)
            if value is not None:
                return value if isinstance(value, str) else str(value)
    except Exception:  # pragma: no cover - unconfigured settings
        pass

    if name in os.environ:
        return os.environ[name]

    try:
        from decouple import config

        return config(name, default=default)
    except Exception:  # pragma: no cover - decouple misconfigured
        return default


@dataclass(frozen=True)
class ProviderSpec:
    """Declarative description of a single generation backend.

    Attributes are deliberately data-only: no network access, no Django
    models, so the registry can be imported from anywhere (forms, admin,
    templates, management commands) without side effects.
    """

    # Identity ----------------------------------------------------------------
    key: str
    """``GenerationJob.engine`` value, e.g. ``"veo"``. Lowercase, <= 20 chars."""
    label: str
    """What the user sees in the engine dropdown."""
    kind: str = "cloud"
    """``"cloud"`` (hosted API) or ``"local"`` (self-hosted backend)."""
    order: int = 100
    """Sort key for the engine dropdown — lower sorts first."""

    # Wiring ------------------------------------------------------------------
    adapter_cls: Any = None
    """Adapter class to register, or ``None`` when the adapter registers
    itself elsewhere (or is not implemented yet)."""
    adapter_name: str = ""
    """Name used to look the adapter up; defaults to :attr:`key`."""

    # Credentials -------------------------------------------------------------
    env_keys: tuple[str, ...] = ()
    """Required environment variables. The first is treated as the API key."""
    optional_env: tuple[str, ...] = ()
    """Optional overrides (base URL, model id, timeouts…)."""
    config_hint: str = ""
    """Actionable message shown when a required variable is missing."""

    # Capabilities ------------------------------------------------------------
    capabilities: frozenset[str] = frozenset()
    max_duration: float | None = None
    """Longest clip the provider accepts, in seconds (``None`` = no limit)."""
    default_model: str = ""
    preset: Mapping[str, Any] = field(default_factory=dict)
    """Field defaults filled in when the user leaves a setting blank."""

    # Presentation ------------------------------------------------------------
    website: str = ""
    docs_url: str = ""
    pricing: str = ""
    estimated_seconds: float = 90.0
    """Rough wall-clock time for one clip, used to animate progress while the
    provider's API reports no percentage."""
    unavailable_note: str = ""
    """Shown when the engine exists but has no adapter implementation."""

    # ------------------------------------------------------------------ query
    @property
    def resolved_adapter_name(self) -> str:
        return self.adapter_name or self.key

    @property
    def api_key_env(self) -> str:
        return self.env_keys[0] if self.env_keys else ""

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    @property
    def missing_env(self) -> list[str]:
        """Required variables that are currently unset or empty."""
        return [name for name in self.env_keys if not read_env(name)]

    @property
    def is_configured(self) -> bool:
        return not self.missing_env

    @property
    def is_enabled(self) -> bool:
        """Whether this provider is switched on by configuration.

        ``VIDEO_PROVIDERS_ENABLED`` (comma separated) whitelists providers;
        ``VIDEO_PROVIDERS_DISABLED`` removes them. An empty whitelist means
        "everything registered", which keeps setup zero-config.
        """
        enabled = {
            item.strip().lower()
            for item in read_env("VIDEO_PROVIDERS_ENABLED", "").split(",")
            if item.strip()
        }
        disabled = {
            item.strip().lower()
            for item in read_env("VIDEO_PROVIDERS_DISABLED", "").split(",")
            if item.strip()
        }
        if self.key in disabled:
            return False
        if enabled and self.key not in enabled:
            return False
        return True

    @property
    def is_ready(self) -> bool:
        """Enabled, implemented and supplied with credentials."""
        if self.kind == "local":
            # A local backend has no API key; it is ready when it is enabled
            # and an adapter exists.
            return self.is_enabled and self.adapter_cls is not None
        return self.is_enabled and self.adapter_cls is not None and self.is_configured

    def status_note(self) -> str:
        """Short, human explanation of why this provider cannot run yet."""
        if not self.is_enabled:
            return "Disabled in configuration."
        if self.adapter_cls is None:
            return self.unavailable_note or "No adapter installed."
        missing = self.missing_env
        if missing:
            return f"Set {', '.join(missing)}."
        return "Ready."

"""
Provider registry — the single place that knows every available backend.

Providers self-register when their module is imported. ``discover()`` imports
each module in :data:`PROVIDER_MODULES` exactly once, so adding or removing a
backend is a one-file change: drop a module in, it appears in the engine
dropdown, the dashboard and the admin; delete it, everything else follows.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

import importlib
import logging

from .spec import MAX_ENGINE_KEY_LENGTH, ProviderSpec

logger = logging.getLogger("cineforge.generator.providers")

# Modules that contribute providers, in the order they should appear before
# the ad-hoc/local ones. Each module calls ``register_provider(...)`` on import.
PROVIDER_MODULES: tuple[str, ...] = (
    "generator.providers.veo",
    "generator.providers.runway",
    "generator.providers.luma",
    "generator.providers.kling",
    "generator.providers.replicate",
    "generator.providers.huggingface",
    "generator.providers.local",
)

_registry: dict[str, ProviderSpec] = {}
_discovered = False


class RegistrationError(Exception):
    """Raised for registry misuse (duplicate or malformed keys)."""


def register_provider(spec: ProviderSpec) -> ProviderSpec:
    """Register a provider spec and, when it carries one, its adapter class."""
    key = spec.key
    if not key or key != key.lower():
        raise RegistrationError(f"Provider keys must be lowercase: {key!r}")
    if len(key) > MAX_ENGINE_KEY_LENGTH:
        raise RegistrationError(
            f"Provider key {key!r} is {len(key)} chars; GenerationJob.engine "
            f"only allows {MAX_ENGINE_KEY_LENGTH}."
        )

    existing = _registry.get(key)
    if existing is not None and existing is not spec:
        raise RegistrationError(f"Provider {key!r} is already registered.")

    if spec.adapter_cls is not None:
        from ..adapters import registry as adapter_registry

        adapter_registry.register(key, spec.adapter_cls)

    _registry[key] = spec
    return spec


def discover(force: bool = False) -> dict[str, ProviderSpec]:
    """Import every provider module once and return the registry."""
    global _discovered
    if _discovered and not force:
        return _registry
    for modname in PROVIDER_MODULES:
        try:
            importlib.import_module(modname)
        except Exception as exc:  # pragma: no cover - a broken plugin
            logger.exception("Could not load provider module %s: %s", modname, exc)
    # ComfyUI's adapter registers itself in adapters_comfyui; make sure that
    # module has been imported before the local specs look it up.
    try:
        importlib.import_module("generator.adapters_comfyui")
        importlib.import_module("generator.adapters_demo")
    except Exception:  # pragma: no cover - optional backends
        logger.debug("Optional adapter modules unavailable", exc_info=True)
    _discovered = True
    return _registry


def _ensure() -> dict[str, ProviderSpec]:
    return _registry if _discovered else discover()


def all_providers() -> list[ProviderSpec]:
    """Every registered provider, in dropdown order."""
    _ensure()
    return sorted(_registry.values(), key=lambda s: (s.order, s.label.lower()))


def get_provider(key: str | None) -> ProviderSpec | None:
    """Look a provider up by ``engine`` value (case-insensitive)."""
    if not key:
        return None
    _ensure()
    return _registry.get(str(key).strip().lower())


def enabled_providers() -> list[ProviderSpec]:
    """Providers that are switched on (configured or not)."""
    return [spec for spec in all_providers() if spec.is_enabled]


def ready_providers() -> list[ProviderSpec]:
    """Providers that could actually run a job right now."""
    return [spec for spec in all_providers() if spec.is_ready]


def provider_keys() -> list[str]:
    _ensure()
    return list(_registry.keys())


def default_model(key: str | None) -> str:
    """Suggested human-readable model name for an engine, or ``""``."""
    spec = get_provider(key)
    return spec.default_model if spec else ""


def preset_for(key: str | None) -> dict:
    """Field defaults for an engine (empty when the provider defines none)."""
    spec = get_provider(key)
    return dict(spec.preset) if spec else {}


def engine_choices(include_unconfigured: bool = True) -> list[tuple[str, str]]:
    """Choices for the engine dropdown.

    Cloud providers that have no API key yet are still listed, but labelled so
    the reason is obvious instead of failing later in the queue.
    """
    choices: list[tuple[str, str]] = []
    for spec in all_providers():
        label = spec.label
        if not spec.is_enabled:
            continue
        if include_unconfigured and spec.kind == "cloud" and not spec.is_configured:
            missing = ", ".join(spec.missing_env)
            label = f"{label} — set {missing}"
        choices.append((spec.key, label))
    return choices


def status_summary() -> dict:
    """Compact provider status for the dashboard / diagnostics pages."""
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "kind": spec.kind,
            "ready": spec.is_ready,
            "enabled": spec.is_enabled,
            "configured": spec.is_configured,
            "missing_env": spec.missing_env,
            "pricing": spec.pricing,
            "website": spec.website,
            "note": spec.status_note(),
        }
        for spec in all_providers()
    ]

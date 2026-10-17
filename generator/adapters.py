"""
Inference adapter layer — modular, replaceable backends.

CineForge AI — Local AI Video Generation Platform.

Adapters translate a ``GenerationJob`` into a concrete call to the installed
inference backend and return the path to the generated video.

This module intentionally keeps the adapter contract simple so a new backend
(ComfyUI, Wan directly, LTX-Video, a test/demo adapter, etc.) can be plugged
in without rewriting the Django application.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger("cineforge.generator.adapters")


class AdapterError(Exception):
    """Base exception for adapter failures."""


class BaseAdapter(ABC):
    """Abstract base for inference adapters."""

    name: str = "base"

    @abstractmethod
    def validate(self, job: Any) -> tuple[bool, str | None, list[str]]:
        """Return (ok, message, warnings) for the job."""
        ...

    @abstractmethod
    def generate(self, job: Any) -> str | Path | None:
        """Run generation and return the path to the output file, or None."""
        ...

    @property
    def can_run_locally(self) -> bool:
        """Whether this adapter can run without a special backend."""
        return False

    @property
    def human_name(self) -> str:
        return self.name

    @property
    def hardware_notes(self) -> list[str]:
        """Notes about hardware/runtime requirements."""
        return []


class AdapterRegistry:
    """Registry of named adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, type[BaseAdapter]] = {}

    def register(self, name: str, adapter: type[BaseAdapter]) -> None:
        self._adapters[name.lower()] = adapter

    def names(self) -> list[str]:
        """Return the names of all registered adapters."""
        return sorted(self._adapters)

    def get(self, name: str) -> BaseAdapter:
        cls = self._adapters.get(name.lower())
        if cls is None:
            from .adapters_demo import TestAdapter

            if name.lower() == "test":
                return TestAdapter()
            raise AdapterError(f"No adapter registered for '{name}'")
        return cls()


# Module-level registry
registry = AdapterRegistry()


def get_adapter(name: str) -> BaseAdapter:
    """Return an adapter instance by name."""
    return registry.get(name)


# Auto-discovery: register adapters from installed app modules
def _discover_adapters() -> None:
    import importlib

    for modname in (
        "generator.adapters_comfyui",
        "generator.adapters_demo",
    ):
        try:
            importlib.import_module(modname)
        except Exception as exc:  # pragma: no cover - optional modules
            logger.debug("Could not import adapter module %s: %s", modname, exc)


_discover_adapters()

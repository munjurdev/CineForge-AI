"""
Cloud video-generation providers.

A provider is a self-contained module that declares a :class:`ProviderSpec`
(what it is, what it needs, what it can do) and a :class:`BaseAdapter`
subclass (how to talk to it). The registry wires both together, so:

* adding a backend  -> drop ``myprovider.py`` into this package and add the
  module name to ``PROVIDER_MODULES``;
* removing a backend -> delete the file (or list its key in
  ``VIDEO_PROVIDERS_DISABLED``).

Nothing outside this package needs to know a provider exists: the engine
dropdown, the dashboard status card, the admin filter and the Celery worker
all read the same registry.

CineForge AI — Local AI Video Generation Platform.
"""
from __future__ import annotations

from .base import CloudVideoAdapter, PollResult, PollState, ProviderError
from .registry import (
    RegistrationError,
    all_providers,
    discover,
    enabled_providers,
    engine_choices,
    get_provider,
    provider_keys,
    ready_providers,
    register_provider,
    status_summary,
)
from .spec import (
    CAP_ASPECT,
    CAP_AUDIO,
    CAP_GUIDANCE,
    CAP_I2V,
    CAP_NEGATIVE,
    CAP_RESOLUTION,
    CAP_SEED,
    CAP_STEPS,
    ProviderSpec,
    read_env,
)

discover()

__all__ = [
    "CAP_ASPECT",
    "CAP_AUDIO",
    "CAP_GUIDANCE",
    "CAP_I2V",
    "CAP_NEGATIVE",
    "CAP_RESOLUTION",
    "CAP_SEED",
    "CAP_STEPS",
    "CloudVideoAdapter",
    "PollResult",
    "PollState",
    "ProviderError",
    "ProviderSpec",
    "RegistrationError",
    "all_providers",
    "discover",
    "enabled_providers",
    "engine_choices",
    "get_provider",
    "provider_keys",
    "ready_providers",
    "read_env",
    "register_provider",
    "status_summary",
]

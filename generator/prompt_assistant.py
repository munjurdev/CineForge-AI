"""
Offline, rule-based prompt assistant.

CineForge AI — Local AI Video Generation Platform.

This module expands a short idea into a structured cinematic prompt using
deterministic templates. It never calls a language model or a remote API, and
callers must not describe its output as AI-enhanced: the result is a rule-based
rewrite that adds framing, lighting, camera movement, and style hints.

The same idea always produces the same prompt so results are reproducible and
easy to test.
"""
from __future__ import annotations

import hashlib

MAX_IDEA_LENGTH = 2000

_SHOTS = (
    "wide establishing shot",
    "medium shot",
    "slow tracking shot",
    "intimate close-up",
)
_LIGHTING = (
    "soft golden-hour light",
    "moody low-key lighting",
    "cool moonlight",
    "natural diffused daylight",
)
_MOVEMENTS = (
    "slow dolly-in",
    "gentle handheld drift",
    "smooth crane rise",
    "lateral tracking move",
)
_STYLES = (
    "35mm film grain",
    "anamorphic lens character",
    "high dynamic range",
    "painterly colour grade",
)

METHOD = "rule-based"


def _pick(options: tuple[str, ...], seed: str, offset: int) -> str:
    """Pick a deterministic option from a tuple for the given seed."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return options[digest[offset] % len(options)]


def build_cinematic_prompt(idea: str) -> str:
    """Return a structured prompt for a short idea.

    Raises ``ValueError`` when the idea is empty or whitespace only.
    """
    cleaned = " ".join((idea or "").split())
    if not cleaned:
        raise ValueError("Write a short idea first.")

    truncated = len(cleaned) > MAX_IDEA_LENGTH
    if truncated:
        cleaned = cleaned[:MAX_IDEA_LENGTH].rstrip()

    shot = _pick(_SHOTS, cleaned, 0)
    lighting = _pick(_LIGHTING, cleaned, 1)
    movement = _pick(_MOVEMENTS, cleaned, 2)
    style = _pick(_STYLES, cleaned, 3)

    sentence = cleaned.rstrip(".!? ")
    prompt = (
        f"{sentence}. {shot.capitalize()}, {lighting}, {movement}, "
        f"{style}, coherent motion, high detail."
    )
    if truncated:
        prompt += " (Shortened from a longer idea.)"
    return prompt

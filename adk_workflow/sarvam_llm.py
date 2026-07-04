"""Sarvam-backed reasoning helper for ADK workflow agents.

This module is intentionally small and model-provider specific. Workflow
agents import `sarvam_reasoning` so they never need to instantiate Gemini or
any Google-hosted LLM for routing decisions.
"""

from __future__ import annotations

import os
from typing import Any

from .sarvam_client import post_chat


DEFAULT_SARVAM_CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-30b").strip()


class SarvamLLMError(RuntimeError):
    """Raised when Sarvam cannot produce a usable reasoning response."""


def sarvam_reasoning(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.05,
    max_tokens: int = 240,
    reasoning_effort: str = "low",
    timeout: int = 60,
) -> str:
    """Return the Sarvam model response for a reasoning-only agent turn.

    Args:
        messages: OpenAI-compatible chat messages.
        temperature: Low values keep routing deterministic.
        max_tokens: Upper bound for the JSON routing decision.
        reasoning_effort: Sarvam reasoning setting passed through unchanged.
        timeout: HTTP request timeout in seconds.

    Returns:
        The model's content, falling back to explicit reasoning fields only
        when Sarvam returns them as the primary textual response.

    Raises:
        SarvamLLMError: If configuration, transport, or response parsing fails.
    """
    try:
        data = post_chat(
            messages=messages,
            model=os.getenv("SARVAM_CHAT_MODEL", DEFAULT_SARVAM_CHAT_MODEL).strip(),
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )
    except Exception as exc:
        raise SarvamLLMError(f"Sarvam request failed: {exc}") from exc

    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        raise SarvamLLMError(f"Sarvam returned no choices: {data}")

    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""

    if isinstance(reasoning, dict):
        reasoning = reasoning.get("content") or reasoning.get("text") or ""

    result = str(content or reasoning).strip()
    if not result:
        raise SarvamLLMError(f"Sarvam returned an empty message: {data}")

    return result

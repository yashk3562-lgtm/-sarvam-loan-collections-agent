"""Sarvam-backed reasoning helper for ADK workflow agents.

This module is intentionally small and model-provider specific. Workflow
agents import `sarvam_reasoning` so they never need to instantiate Gemini or
any Google-hosted LLM for routing decisions.
"""

from __future__ import annotations

import os
from typing import Any


DEFAULT_SARVAM_CHAT_URL = os.getenv(
    "SARVAM_CHAT_URL",
    "https://api.sarvam.ai/v1/chat/completions",
).strip()
DEFAULT_SARVAM_CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-30b").strip()


class SarvamLLMError(RuntimeError):
    """Raised when Sarvam cannot produce a usable reasoning response."""


def _sarvam_headers() -> dict[str, str]:
    """Build authenticated headers for Sarvam chat completions."""
    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not api_key:
        raise SarvamLLMError("Missing SARVAM_API_KEY environment variable.")

    return {
        "api-subscription-key": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _response_error(response: Any) -> str:
    """Extract a compact error message from a Sarvam API response."""
    try:
        payload = response.json()
    except ValueError:
        return response.text

    if isinstance(payload, dict):
        return str(payload.get("message") or payload.get("error") or payload)
    return str(payload)


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
    payload: dict[str, Any] = {
        "model": os.getenv("SARVAM_CHAT_MODEL", DEFAULT_SARVAM_CHAT_MODEL).strip(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }

    try:
        import requests

        response = requests.post(
            os.getenv("SARVAM_CHAT_URL", DEFAULT_SARVAM_CHAT_URL).strip(),
            headers=_sarvam_headers(),
            json=payload,
            timeout=timeout,
        )
    except ImportError as exc:
        raise SarvamLLMError(
            "The requests package is required for Sarvam reasoning."
        ) from exc
    except Exception as exc:
        raise SarvamLLMError(f"Sarvam request failed: {exc}") from exc

    if response.status_code >= 400:
        raise SarvamLLMError(
            f"Sarvam chat failed with HTTP {response.status_code}: {_response_error(response)}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise SarvamLLMError("Sarvam returned a non-JSON response.") from exc

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

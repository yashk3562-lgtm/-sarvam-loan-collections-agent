"""Shared Sarvam API helpers for the Sarvam collections agent repo."""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

import requests

DEFAULT_SARVAM_CHAT_URL = os.getenv(
    "SARVAM_CHAT_URL",
    "https://api.sarvam.ai/v1/chat/completions",
).strip()
DEFAULT_SARVAM_STT_URL = os.getenv(
    "SARVAM_STT_URL",
    "https://api.sarvam.ai/speech-to-text",
).strip()
DEFAULT_SARVAM_TTS_URL = os.getenv(
    "SARVAM_TTS_URL",
    "https://api.sarvam.ai/text-to-speech",
).strip()
DEFAULT_SARVAM_CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-30b").strip()
DEFAULT_SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3").strip()
DEFAULT_SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3").strip()


class SarvamAPIError(RuntimeError):
    """Raised when a Sarvam API request fails."""


def get_sarvam_api_key(api_key: str | None = None) -> str:
    if api_key and api_key.strip():
        return api_key.strip()

    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        raise SarvamAPIError("Missing SARVAM_API_KEY environment variable.")
    return key


def sarvam_headers(json_mode: bool = True, api_key: str | None = None) -> dict[str, str]:
    key = get_sarvam_api_key(api_key)
    headers = {
        "api-subscription-key": key,
        "Authorization": f"Bearer {key}",
    }
    if json_mode:
        headers["Content-Type"] = "application/json"
    return headers


def _response_error(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text

    if isinstance(payload, dict):
        return str(payload.get("message") or payload.get("error") or payload)
    return str(payload)


def post_chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.05,
    max_tokens: int = 240,
    reasoning_effort: str = "low",
    timeout: int = 60,
    api_key: str | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model or DEFAULT_SARVAM_CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }
    response = requests.post(
        url or DEFAULT_SARVAM_CHAT_URL,
        headers=sarvam_headers(True, api_key=api_key),
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise SarvamAPIError(
            f"Sarvam chat failed with HTTP {response.status_code}: {_response_error(response)}"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise SarvamAPIError("Sarvam returned a non-JSON chat response.") from exc


def post_stt(
    audio_file: Any,
    model: str | None = None,
    mode: str = "codemix",
    timeout: int = 90,
    api_key: str | None = None,
    url: str | None = None,
) -> str:
    response = requests.post(
        url or DEFAULT_SARVAM_STT_URL,
        headers=sarvam_headers(False, api_key=api_key),
        files={"file": ("borrower_response.wav", audio_file.getvalue(), "audio/wav")},
        data={"model": model or DEFAULT_SARVAM_STT_MODEL, "mode": mode},
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise SarvamAPIError(
            f"Sarvam STT failed with HTTP {response.status_code}: {_response_error(response)}"
        )

    payload = response.json()
    return (
        payload.get("transcript")
        or payload.get("text")
        or payload.get("transcription")
        or ""
    ).strip()


def post_tts(
    text: str,
    target_language_code: str,
    speaker: str,
    model: str | None = None,
    pace: float = 1.08,
    speech_sample_rate: int = 24000,
    output_audio_codec: str = "wav",
    temperature: float = 0.3,
    timeout: int = 90,
    api_key: str | None = None,
    url: str | None = None,
) -> bytes | None:
    response = requests.post(
        url or DEFAULT_SARVAM_TTS_URL,
        headers=sarvam_headers(True, api_key=api_key),
        json={
            "text": text[:1000],
            "target_language_code": target_language_code,
            "speaker": speaker,
            "model": model or DEFAULT_SARVAM_TTS_MODEL,
            "pace": pace,
            "speech_sample_rate": speech_sample_rate,
            "output_audio_codec": output_audio_codec,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise SarvamAPIError(
            f"Sarvam TTS failed with HTTP {response.status_code}: {_response_error(response)}"
        )

    payload = response.json()
    audios = payload.get("audios") or []
    if not audios:
        return None
    return base64.b64decode(audios[0])


def extract_json_object(text: str) -> dict[str, Any]:
    if not text or not isinstance(text, str):
        raise ValueError("Empty JSON response")

    cleaned = text.strip()
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace("‘", "'").replace("’", "'")
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {text[:300]}")

    candidate = cleaned[start : end + 1]
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
    candidate = re.sub(r"(?<=[{,])\s*'([^'{}:\[\],]+)'\s*:", r'"\1":', candidate)
    candidate = re.sub(
        r":\s*'([^'\\]*(?:\\.[^'\\]*)*)'",
        lambda match: ": " + json.dumps(match.group(1)),
        candidate,
    )
    candidate = re.sub(r"\bTrue\b", "true", candidate)
    candidate = re.sub(r"\bFalse\b", "false", candidate)
    candidate = re.sub(r"\bNone\b", "null", candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON response: {exc}") from exc


__all__ = [
    "DEFAULT_SARVAM_CHAT_MODEL",
    "DEFAULT_SARVAM_STT_MODEL",
    "DEFAULT_SARVAM_TTS_MODEL",
    "SarvamAPIError",
    "extract_json_object",
    "get_sarvam_api_key",
    "post_chat",
    "post_stt",
    "post_tts",
    "sarvam_headers",
]

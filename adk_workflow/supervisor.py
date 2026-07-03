"""Sarvam-powered Google ADK supervisor for collections routing.

The supervisor is deliberately narrow: it never speaks to the borrower and
does not execute collections business logic. It asks Sarvam to choose the next
specialist agent, validates the response, and emits a JSON routing decision.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

try:  # ADK is available in the production runtime via google-adk.
    from google.adk.agents import BaseAgent
    from google.adk.events import Event, EventActions
    from google.genai import types

    ADK_AVAILABLE = True
except ImportError:  # Keeps direct unit tests importable before deps install.
    ADK_AVAILABLE = False

    class BaseAgent:  # type: ignore[no-redef]
        """Minimal import-time fallback when google-adk is not installed."""

        def __init__(self, name: str, description: str = "", **_: Any) -> None:
            self.name = name
            self.description = description

    Event = None  # type: ignore[assignment]
    EventActions = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

from .sarvam_llm import SarvamLLMError, sarvam_reasoning


ALLOWED_NEXT_AGENTS = frozenset({"negotiation", "hardship"})

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor Agent for a loan collections workflow.

You never answer the borrower directly.
You never generate borrower-facing text.
You perform no collections business action.
You only decide which specialist agent should execute next.

Allowed next_agent values:
- negotiation
- hardship

Do not route to compliance.
Do not route to CRM.
Compliance always runs after the selected business specialist.
CRM tools run later only for final outcomes.

Routing rules:

Negotiation:
- borrower willing to negotiate
- partial payment
- promise-to-pay
- payment date
- salary date
- alternate payment plan

Hardship:
- lost job
- medical issue
- financial hardship
- death in family
- temporary inability

Return only valid JSON with this exact schema:
{
  "next_agent": "negotiation|hardship",
  "reason": "brief routing rationale",
  "confidence": 0.0
}

The confidence must be a number from 0.0 to 1.0.

Output rules:
- Return ONLY one JSON object.
- No markdown.
- No explanation.
- No reasoning.
- No numbered analysis.
- No placeholder values like "...".
- First character must be {
- Last character must be }
- next_agent must be exactly one of:
  negotiation
  hardship
"""


def _json_default(value: Any) -> str:
    """Provide stable serialization for non-JSON-native profile values."""
    return str(value)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a potentially messy model response."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Sarvam returned an empty routing response.")

    cleaned = text.strip()
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in routing response: {text[:300]}")

    candidate = cleaned[start : end + 1]
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
    candidate = re.sub(r"\bTrue\b", "true", candidate)
    candidate = re.sub(r"\bFalse\b", "false", candidate)
    candidate = re.sub(r"\bNone\b", "null", candidate)
    return json.loads(candidate)


def _coerce_confidence(value: Any) -> float:
    """Normalize confidence to the required 0.0-1.0 range."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.7
    return max(0.0, min(1.0, confidence))


def _normalized_outcome(value: Any) -> str:
    """Normalize final outcome labels for fallback routing."""
    if isinstance(value, dict):
        for key in ("final_status", "status", "outcome", "recommended_action"):
            if value.get(key):
                return _normalized_outcome(value[key])
        return " ".join(_normalized_outcome(item) for item in value.values()).strip()
    if isinstance(value, list):
        return " ".join(_normalized_outcome(item) for item in value).strip()
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _fallback_next_agent(
    borrower_state: dict[str, Any] | None = None,
    final_outcome: Any = None,
) -> str:
    """Choose a valid next agent when Sarvam returns an invalid value."""
    state = borrower_state or {}

    if state.get("financial_hardship") is True or str(
        state.get("job_status", "")
    ).strip().lower() == "lost_job":
        return "hardship"
    return "negotiation"


def normalize_supervisor_decision(
    payload: dict[str, Any],
    borrower_state: dict[str, Any] | None = None,
    final_outcome: Any = None,
) -> dict[str, Any]:
    """Canonicalize a supervisor decision without raising for invalid agents."""
    next_agent = str(payload.get("next_agent", "")).strip().lower()
    if next_agent not in ALLOWED_NEXT_AGENTS:
        next_agent = _fallback_next_agent(
            borrower_state=borrower_state,
            final_outcome=final_outcome,
        )

    reason = str(payload.get("reason", "")).strip()
    if not reason or reason == "...":
        reason = "Fallback routing based on borrower state and final outcome."

    return {
        "next_agent": next_agent,
        "reason": reason,
        "confidence": _coerce_confidence(payload.get("confidence", 0.7)),
    }


def _normalize_decision(
    payload: dict[str, Any],
    borrower_state: dict[str, Any] | None = None,
    final_outcome: Any = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper for supervisor decision normalization."""
    return normalize_supervisor_decision(
        payload,
        borrower_state=borrower_state,
        final_outcome=final_outcome,
    )


def _build_messages(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
    borrower_state: dict[str, Any] | None = None,
    final_outcome: Any = None,
) -> list[dict[str, str]]:
    """Create the Sarvam prompt for a single routing decision."""
    routing_input = {
        "borrower_profile": borrower_profile,
        "conversation_history": conversation_history,
        "latest_borrower_message": latest_borrower_message,
        "borrower_state": borrower_state or {},
        "final_outcome": final_outcome,
    }

    return [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Route this collections conversation to the next specialist agent. "
                "Return ONLY one JSON object. No markdown, explanation, reasoning, "
                'numbered analysis, or placeholder values like "...". First '
                "character must be { and last character must be }. next_agent "
                "must be exactly one of negotiation, hardship. Do not route to "
                "compliance or CRM.\n\n"
                + json.dumps(routing_input, ensure_ascii=False, default=_json_default)
            ),
        },
    ]


def decide_next_agent(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
    borrower_state: dict[str, Any] | None = None,
    final_outcome: Any = None,
) -> dict[str, Any]:
    """Ask Sarvam to choose the next specialist agent."""
    raw = sarvam_reasoning(
        _build_messages(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_message=latest_borrower_message,
            borrower_state=borrower_state,
            final_outcome=final_outcome,
        )
    )
    return normalize_supervisor_decision(
        _extract_json_object(raw),
        borrower_state=borrower_state,
        final_outcome=final_outcome,
    )


def decide_next_agent_json(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
    borrower_state: dict[str, Any] | None = None,
    final_outcome: Any = None,
) -> str:
    """Return a compact JSON string for the supervisor routing decision."""
    return json.dumps(
        decide_next_agent(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_message=latest_borrower_message,
            borrower_state=borrower_state,
            final_outcome=final_outcome,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _content_text(content: Any) -> str:
    """Extract text from an ADK/GenAI content object."""
    if content is None:
        return ""

    parts = getattr(content, "parts", None) or []
    texts = [str(getattr(part, "text", "") or "") for part in parts]
    text = "\n".join(part for part in texts if part).strip()
    return text or str(content)


def _context_payload(
    parent_context: Any,
) -> tuple[dict[str, Any], Any, str, dict[str, Any] | None, Any]:
    """Read supervisor inputs from ADK session state and user content."""
    session = getattr(parent_context, "session", None)
    state = getattr(session, "state", {}) or {}
    latest_text = _content_text(getattr(parent_context, "user_content", None))

    borrower_profile = dict(
        state.get("borrower_profile")
        or state.get("borrower")
        or state.get("account")
        or {}
    )
    conversation_history = state.get("conversation_history") or state.get("messages") or []
    latest_borrower_message = state.get("latest_borrower_message") or latest_text
    borrower_state = state.get("borrower_state") or state.get("structured_borrower_state")
    final_outcome = state.get("final_outcome") or state.get("final_status")

    try:
        payload = json.loads(latest_text) if latest_text else {}
    except json.JSONDecodeError:
        payload = {}

    if isinstance(payload, dict):
        borrower_profile = dict(payload.get("borrower_profile") or borrower_profile)
        conversation_history = payload.get("conversation_history", conversation_history)
        latest_borrower_message = str(
            payload.get("latest_borrower_message") or latest_borrower_message
        )
        payload_state = (
            payload.get("borrower_state")
            or payload.get("structured_borrower_state")
            or borrower_state
        )
        borrower_state = payload_state if isinstance(payload_state, dict) else None
        final_outcome = (
            payload.get("final_outcome")
            or payload.get("final_status")
            or final_outcome
        )

    return (
        borrower_profile,
        conversation_history,
        latest_borrower_message,
        borrower_state if isinstance(borrower_state, dict) else None,
        final_outcome,
    )


def _adk_event(parent_context: Any, author: str, text: str) -> Any:
    """Build an ADK Event containing the supervisor JSON response."""
    if not ADK_AVAILABLE:
        return text

    return Event(
        author=author,
        invocation_id=getattr(parent_context, "invocation_id", ""),
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        actions=EventActions(),
        timestamp=time.time(),
    )


class SarvamSupervisorAgent(BaseAgent):
    """ADK supervisor agent that delegates routing intelligence to Sarvam."""

    async def _run_async_impl(self, parent_context: Any):
        """Run one text-routing turn and yield a JSON-only ADK event."""
        (
            borrower_profile,
            conversation_history,
            latest_message,
            borrower_state,
            final_outcome,
        ) = _context_payload(parent_context)
        decision = decide_next_agent_json(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_message=latest_message,
            borrower_state=borrower_state,
            final_outcome=final_outcome,
        )
        yield _adk_event(parent_context, self.name, decision)

    async def _run_live_impl(self, parent_context: Any):
        """Route live invocations using the same JSON-only supervisor path."""
        async for event in self._run_async_impl(parent_context):
            yield event


root_agent = SarvamSupervisorAgent(
    name="supervisor",
    description=(
        "Routes collections conversations to negotiation or hardship specialists "
        "using Sarvam reasoning."
    ),
)


__all__ = [
    "ALLOWED_NEXT_AGENTS",
    "SarvamSupervisorAgent",
    "decide_next_agent",
    "decide_next_agent_json",
    "normalize_supervisor_decision",
    "root_agent",
]

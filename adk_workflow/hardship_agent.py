"""Sarvam-powered Google ADK hardship specialist for collections.

This agent does not route, update CRM, run compliance checks, or generate
borrower-facing speech. Its only job is to reason about financial hardship and
return a strict JSON decision describing what should happen next.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

try:  # google-adk is provided by the production runtime.
    from google.adk.agents import BaseAgent
    from google.adk.events import Event, EventActions
    from google.genai import types

    ADK_AVAILABLE = True
except ImportError:  # Keeps local imports/test doubles working before deps install.
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


ALLOWED_STATUSES = frozenset({"continue", "promise_to_pay", "escalate"})

HARDSHIP_SYSTEM_PROMPT = """You are the Hardship Specialist Agent for a loan collections workflow.

Scope:
- Reason only about financial hardship.
- Do not perform routing.
- Do not perform CRM updates.
- Do not perform compliance checks.
- Do not generate borrower-facing spoken responses.
- Return only JSON describing what should happen next.

Input:
- borrower profile
- full conversation history
- latest borrower utterance

Hardship signals include:
- job loss
- salary delay
- medical emergency
- accident
- business loss
- family emergency
- temporary cash flow issue

Reasoning requirements:
- Read the full conversation history before deciding.
- If the borrower mentions a hardship signal, empathize internally and continue exploring.
- Decide intelligently whether the next question should ask about expected income date, salary date, partial payment, family support, employer settlement, government benefits, any affordable amount, or callback timing.
- Never ask about salary if the borrower explicitly says they lost their job.
- Never ask the same question twice. Compare the next_question against the full history.
- If the borrower eventually provides a payment date, set status to "promise_to_pay".
- If the borrower cannot provide any date after multiple attempts and cannot make partial payment, set status to "escalate".
- Otherwise set status to "continue".

Strict output schema:
{
  "status": "continue|promise_to_pay|escalate",
  "reason": "brief internal rationale",
  "next_question": "single next question for the next specialist to ask, or empty string when no question is needed",
  "summary": "concise hardship summary"
}

Return only valid JSON. No markdown. No labels. No spoken borrower response.
"""


def _json_default(value: Any) -> str:
    """Provide stable serialization for non-JSON-native profile values."""
    return str(value)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from a Sarvam response."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Sarvam returned an empty hardship response.")

    cleaned = text.strip()
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in hardship response: {text[:300]}")

    candidate = re.sub(r",(\s*[}\]])", r"\1", cleaned[start : end + 1])
    return json.loads(candidate)


def _normalize_text(value: Any) -> str:
    """Normalize required JSON string fields."""
    if value is None:
        return ""
    return str(value).strip()


def _history_text(
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_utterance: str,
) -> str:
    """Flatten conversation context for local safety checks."""
    if isinstance(conversation_history, str):
        history = conversation_history
    else:
        history = json.dumps(
            conversation_history,
            ensure_ascii=False,
            default=_json_default,
        )
    return f"{history}\n{latest_borrower_utterance}".lower()


def _mentions_job_loss(context: str) -> bool:
    """Return whether the borrower explicitly described job loss."""
    job_loss_terms = (
        "lost my job",
        "lost job",
        "job loss",
        "no job",
        "unemployed",
        "laid off",
        "terminated",
        "fired",
        "company removed me",
        " नौकरी चली",
        "नौकरी चली",
        "job chali",
        "naukri chali",
    )
    return any(term in context for term in job_loss_terms)


def _question_was_already_asked(context: str, next_question: str) -> bool:
    """Detect exact repeated questions in the prior conversation."""
    normalized_question = re.sub(r"\s+", " ", next_question.lower()).strip()
    normalized_context = re.sub(r"\s+", " ", context).strip()
    return bool(normalized_question and normalized_question in normalized_context)


def _safe_replacement_question(context: str) -> str:
    """Choose a non-repeated hardship exploration question."""
    candidates = [
        "Is there any affordable amount you can arrange right now?",
        "Can anyone in your family support a small partial payment?",
        "Are you expecting any employer settlement or other benefit soon?",
        "What callback timing would work best to review this again?",
    ]
    for candidate in candidates:
        if not _question_was_already_asked(context, candidate):
            return candidate
    return ""


def _apply_hardship_guardrails(
    decision: dict[str, str],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_utterance: str,
) -> dict[str, str]:
    """Enforce non-negotiable hardship questioning constraints."""
    if decision["status"] != "continue":
        return decision

    context = _history_text(conversation_history, latest_borrower_utterance)
    question_lower = decision["next_question"].lower()
    asks_salary = "salary" in question_lower or "payday" in question_lower

    if (_mentions_job_loss(context) and asks_salary) or _question_was_already_asked(
        context, decision["next_question"]
    ):
        decision = dict(decision)
        decision["next_question"] = _safe_replacement_question(context)
        if not decision["next_question"]:
            decision["status"] = "escalate"
            decision["reason"] = (
                "No non-repeated hardship exploration question remains after prior attempts."
            )

    return decision


def _normalize_decision(payload: dict[str, Any]) -> dict[str, str]:
    """Validate and canonicalize a hardship decision."""
    status = _normalize_text(payload.get("status")).lower()
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Invalid hardship status: {status!r}")

    reason = _normalize_text(payload.get("reason"))
    summary = _normalize_text(payload.get("summary"))
    next_question = _normalize_text(payload.get("next_question"))

    if not reason:
        raise ValueError("Hardship decision is missing a reason.")
    if not summary:
        raise ValueError("Hardship decision is missing a summary.")
    if status == "continue" and not next_question:
        raise ValueError("Hardship decision must include next_question when continuing.")

    normalized = {
        "status": status,
        "reason": reason,
        "next_question": next_question,
        "summary": summary,
    }
    return normalized


def _build_messages(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_utterance: str,
) -> list[dict[str, str]]:
    """Create the Sarvam prompt for a single hardship reasoning turn."""
    hardship_input = {
        "borrower_profile": borrower_profile,
        "conversation_history": conversation_history,
        "latest_borrower_utterance": latest_borrower_utterance,
    }

    return [
        {"role": "system", "content": HARDSHIP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Analyze this hardship context. Return strict JSON only.\n\n"
                + json.dumps(hardship_input, ensure_ascii=False, default=_json_default)
            ),
        },
    ]


def analyze_hardship(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_utterance: str,
) -> dict[str, str]:
    """Ask Sarvam to produce a strict hardship-only workflow decision.

    Args:
        borrower_profile: Borrower/account metadata available to the workflow.
        conversation_history: Full prior conversation turns.
        latest_borrower_utterance: The newest borrower utterance.

    Returns:
        A validated decision with `status`, `reason`, `next_question`, and
        `summary`.

    Raises:
        SarvamLLMError: If Sarvam cannot be reached or returns no content.
        ValueError: If Sarvam returns malformed or invalid hardship JSON.
    """
    raw = sarvam_reasoning(
        _build_messages(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_utterance=latest_borrower_utterance,
        )
    )
    decision = _normalize_decision(_extract_json_object(raw))
    return _apply_hardship_guardrails(
        decision=decision,
        conversation_history=conversation_history,
        latest_borrower_utterance=latest_borrower_utterance,
    )


def analyze_hardship_json(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_utterance: str,
) -> str:
    """Return a compact JSON string for the hardship decision."""
    return json.dumps(
        analyze_hardship(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_utterance=latest_borrower_utterance,
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


def _context_payload(parent_context: Any) -> tuple[dict[str, Any], Any, str]:
    """Read hardship inputs from ADK session state and user content."""
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
    latest_utterance = (
        state.get("latest_borrower_utterance")
        or state.get("latest_borrower_message")
        or latest_text
    )

    try:
        payload = json.loads(latest_text) if latest_text else {}
    except json.JSONDecodeError:
        payload = {}

    if isinstance(payload, dict):
        borrower_profile = dict(payload.get("borrower_profile") or borrower_profile)
        conversation_history = payload.get("conversation_history", conversation_history)
        latest_utterance = str(
            payload.get("latest_borrower_utterance")
            or payload.get("latest_borrower_message")
            or latest_utterance
        )

    return borrower_profile, conversation_history, latest_utterance


def _adk_event(parent_context: Any, author: str, text: str) -> Any:
    """Build an ADK Event containing the hardship JSON response."""
    if not ADK_AVAILABLE:
        return text

    return Event(
        author=author,
        invocation_id=getattr(parent_context, "invocation_id", ""),
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        actions=EventActions(),
        timestamp=time.time(),
    )


class HardshipAgent(BaseAgent):
    """ADK specialist agent that reasons only about borrower hardship."""

    async def _run_async_impl(self, parent_context: Any):
        """Run one hardship reasoning turn and yield a JSON-only ADK event."""
        borrower_profile, conversation_history, latest_utterance = _context_payload(
            parent_context
        )
        decision = analyze_hardship_json(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_utterance=latest_utterance,
        )
        yield _adk_event(parent_context, self.name, decision)

    async def _run_live_impl(self, parent_context: Any):
        """Handle live invocations through the same hardship JSON path."""
        async for event in self._run_async_impl(parent_context):
            yield event


root_agent = HardshipAgent(
    name="hardship_agent",
    description=(
        "Analyzes borrower financial hardship and returns strict JSON with "
        "continue, promise_to_pay, or escalate status."
    ),
)


__all__ = [
    "HardshipAgent",
    "analyze_hardship",
    "analyze_hardship_json",
    "root_agent",
]

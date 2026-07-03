"""Sarvam-powered Google ADK payment negotiation specialist.

This agent is deliberately narrow: it does not route, update CRM, check
compliance, or produce borrower-facing speech. It reasons only about payment
negotiation and returns a strict JSON workflow decision.
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

NEGOTIATION_SYSTEM_PROMPT = """You are the Payment Negotiation Specialist Agent for a loan collections workflow.

Scope:
- Reason only about payment negotiation.
- Do not perform routing.
- Do not perform CRM updates.
- Do not perform compliance checks.
- Do not generate borrower-facing spoken responses.
- Return only JSON describing what should happen next.

Input:
- borrower profile
- full conversation history
- latest borrower message

Negotiation strategy, in priority order:
1. First determine whether the borrower can pay today.
2. If not today, ask for a realistic payment date.
3. If the borrower cannot provide a date, explore partial payment.
4. If partial payment is impossible, ask about expected future income.
5. If the borrower lost their job, do not ask salary date. Instead explore settlement, savings, family support, government benefits, and callback timing.
6. If the borrower eventually provides a payment date, set status to "promise_to_pay".
7. If after multiple attempts the borrower cannot provide a payment date, any partial payment, or an expected income timeline, set status to "escalate".
8. Otherwise set status to "continue".

Reasoning requirements:
- Read the full conversation history before deciding.
- Reason naturally from what the borrower has already said.
- Never repeat questions. Compare next_question against the full history.
- Never contradict borrower context.
- Never ask about salary date if the borrower explicitly says they lost their job.
- If status is "promise_to_pay" or "escalate", next_question must be an empty string.

Strict output schema:
{
  "status": "continue|promise_to_pay|escalate",
  "reason": "brief internal rationale",
  "next_question": "single next question for the next specialist to ask, or empty string when no question is needed",
  "summary": "concise negotiation summary"
}

Output rules:
- Return ONLY one JSON object.
- No markdown.
- No explanation.
- No reasoning.
- No numbered analysis.
- First character must be {
- Last character must be }
- No labels. No spoken borrower response.
"""

FALLBACK_NEGOTIATION_DECISION = {
    "status": "continue",
    "reason": "Fallback used because Sarvam did not return clean JSON.",
    "next_question": "Ask borrower for earliest realistic payment option.",
    "summary": "Negotiation agent fallback.",
}


def _json_default(value: Any) -> str:
    """Provide stable serialization for non-JSON-native profile values."""
    return str(value)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a potentially messy Sarvam response."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Sarvam returned an empty negotiation response.")

    cleaned = text.strip()
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in negotiation response: {text[:300]}")

    candidate = cleaned[start : end + 1]
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
    candidate = re.sub(r"\bTrue\b", "true", candidate)
    candidate = re.sub(r"\bFalse\b", "false", candidate)
    candidate = re.sub(r"\bNone\b", "null", candidate)
    return json.loads(candidate)


def _normalize_text(value: Any) -> str:
    """Normalize required JSON string fields."""
    if value is None:
        return ""
    return str(value).strip()


def _history_text(
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
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
    return f"{history}\n{latest_borrower_message}".lower()


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
    """Choose a non-repeated negotiation question aligned to the priority ladder."""
    if _mentions_job_loss(context):
        candidates = [
            "Are you expecting any settlement, savings access, family support, or government benefit that could help with payment?",
            "Can anyone in your family support a small partial payment?",
            "Is there any amount from savings you can arrange now?",
            "What callback timing would work best to review your options again?",
        ]
    else:
        candidates = [
            "Can you make the payment today?",
            "If payment is not possible today, what realistic date can you commit to?",
            "Is there any partial amount you can pay before that date?",
            "When do you expect your next income or funds to become available?",
        ]

    for candidate in candidates:
        if not _question_was_already_asked(context, candidate):
            return candidate
    return ""


def _normalize_decision(payload: dict[str, Any]) -> dict[str, str]:
    """Validate and canonicalize a negotiation decision."""
    status = _normalize_text(payload.get("status")).lower()
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Invalid negotiation status: {status!r}")

    reason = _normalize_text(payload.get("reason"))
    summary = _normalize_text(payload.get("summary"))
    next_question = _normalize_text(payload.get("next_question"))

    if not reason:
        raise ValueError("Negotiation decision is missing a reason.")
    if not summary:
        raise ValueError("Negotiation decision is missing a summary.")
    if status == "continue" and not next_question:
        raise ValueError("Negotiation decision must include next_question when continuing.")

    return {
        "status": status,
        "reason": reason,
        "next_question": next_question if status == "continue" else "",
        "summary": summary,
    }


def _apply_negotiation_guardrails(
    decision: dict[str, str],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
) -> dict[str, str]:
    """Enforce non-negotiable negotiation questioning constraints."""
    if decision["status"] != "continue":
        return decision

    context = _history_text(conversation_history, latest_borrower_message)
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
                "No non-repeated payment negotiation question remains after prior attempts."
            )

    return decision


def _build_messages(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
) -> list[dict[str, str]]:
    """Create the Sarvam prompt for a single negotiation reasoning turn."""
    negotiation_input = {
        "borrower_profile": borrower_profile,
        "conversation_history": conversation_history,
        "latest_borrower_message": latest_borrower_message,
    }

    return [
        {"role": "system", "content": NEGOTIATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Analyze this payment negotiation context. Return ONLY one JSON "
                "object. First character must be { and last character must be }. "
                "No markdown, explanation, reasoning, or numbered analysis.\n\n"
                + json.dumps(negotiation_input, ensure_ascii=False, default=_json_default)
            ),
        },
    ]


def _repair_messages(raw_output: str) -> list[dict[str, str]]:
    """Create a one-shot JSON repair prompt for malformed Sarvam output."""
    required_schema = {
        "status": "continue|promise_to_pay|escalate",
        "reason": "brief internal rationale",
        "next_question": (
            "single next question for the next specialist to ask, or empty "
            "string when no question is needed"
        ),
        "summary": "concise negotiation summary",
    }

    return [
        {
            "role": "system",
            "content": (
                "You repair malformed model output into valid JSON. Return ONLY "
                "one JSON object. No markdown. No explanation. First character "
                "must be { and last character must be }."
            ),
        },
        {
            "role": "user",
            "content": (
                "Convert this raw output into valid JSON using the required schema. "
                "Return JSON only.\n\nRequired schema:\n"
                + json.dumps(required_schema, ensure_ascii=False, default=_json_default)
                + "\n\nRaw output:\n"
                + raw_output
            ),
        },
    ]


def _fallback_decision() -> dict[str, str]:
    """Return the safe fallback negotiation decision."""
    return dict(FALLBACK_NEGOTIATION_DECISION)


def _parse_or_repair_negotiation_decision(raw: str) -> dict[str, str]:
    """Parse negotiation JSON, retrying Sarvam once for repair if needed."""
    try:
        return _normalize_decision(_extract_json_object(raw))
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        repaired = sarvam_reasoning(
            _repair_messages(raw),
            temperature=0.0,
            max_tokens=240,
        )
        return _normalize_decision(_extract_json_object(repaired))
    except (SarvamLLMError, json.JSONDecodeError, ValueError):
        return _fallback_decision()


def negotiate_payment(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
) -> dict[str, str]:
    """Ask Sarvam to produce a strict payment-negotiation workflow decision.

    Args:
        borrower_profile: Borrower/account metadata available to the workflow.
        conversation_history: Full prior conversation turns.
        latest_borrower_message: The newest borrower utterance.

    Returns:
        A validated decision with `status`, `reason`, `next_question`, and
        `summary`.

    Raises:
        SarvamLLMError: If Sarvam cannot be reached or returns no content.
        ValueError: If Sarvam returns malformed or invalid negotiation JSON.
    """
    raw = sarvam_reasoning(
        _build_messages(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_message=latest_borrower_message,
        )
    )
    decision = _parse_or_repair_negotiation_decision(raw)
    return _apply_negotiation_guardrails(
        decision=decision,
        conversation_history=conversation_history,
        latest_borrower_message=latest_borrower_message,
    )


def negotiate_payment_json(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
) -> str:
    """Return a compact JSON string for the negotiation decision."""
    return json.dumps(
        negotiate_payment(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_message=latest_borrower_message,
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
    """Read negotiation inputs from ADK session state and user content."""
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

    return borrower_profile, conversation_history, latest_borrower_message


def _adk_event(parent_context: Any, author: str, text: str) -> Any:
    """Build an ADK Event containing the negotiation JSON response."""
    if not ADK_AVAILABLE:
        return text

    return Event(
        author=author,
        invocation_id=getattr(parent_context, "invocation_id", ""),
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        actions=EventActions(),
        timestamp=time.time(),
    )


class NegotiationAgent(BaseAgent):
    """ADK specialist agent that reasons only about payment negotiation."""

    async def _run_async_impl(self, parent_context: Any):
        """Run one negotiation reasoning turn and yield a JSON-only ADK event."""
        borrower_profile, conversation_history, latest_message = _context_payload(
            parent_context
        )
        decision = negotiate_payment_json(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_message=latest_message,
        )
        yield _adk_event(parent_context, self.name, decision)

    async def _run_live_impl(self, parent_context: Any):
        """Handle live invocations through the same negotiation JSON path."""
        async for event in self._run_async_impl(parent_context):
            yield event


root_agent = NegotiationAgent(
    name="negotiation_agent",
    description=(
        "Analyzes borrower payment negotiation context and returns strict JSON "
        "with continue, promise_to_pay, or escalate status."
    ),
)


__all__ = [
    "NegotiationAgent",
    "negotiate_payment",
    "negotiate_payment_json",
    "root_agent",
]

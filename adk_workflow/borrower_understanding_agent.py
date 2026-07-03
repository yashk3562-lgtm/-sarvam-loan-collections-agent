"""Sarvam-powered Google ADK borrower understanding specialist.

This agent only converts messy borrower conversation into structured borrower
state. It does not route, negotiate, update CRM, create reminders, or produce
borrower-facing speech.
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


ALLOWED_INTENTS = frozenset(
    {
        "can_pay",
        "cannot_pay",
        "partial_payment",
        "promise_to_pay",
        "hardship",
        "dispute",
        "callback_request",
        "unknown",
    }
)
ALLOWED_JOB_STATUSES = frozenset({"employed", "lost_job", "unknown"})

BORROWER_UNDERSTANDING_SYSTEM_PROMPT = """You extract structured borrower state from Indian loan collection conversations.
Return only one valid JSON object.
Do not explain.
Do not include markdown.
The first character must be { and the last character must be }.
"""

FALLBACK_BORROWER_STATE = {
    "borrower_name": "",
    "intent": "unknown",
    "today_payment": False,
    "promise_to_pay": False,
    "payment_date": "",
    "partial_payment_possible": False,
    "partial_payment_amount": "",
    "job_status": "unknown",
    "income_date": "",
    "financial_hardship": False,
    "callback_requested": False,
    "human_escalation_requested": False,
    "dispute_or_fraud": False,
    "confidence": 0.3,
    "evidence": "Fallback used because Sarvam did not return clean JSON.",
}


def _json_default(value: Any) -> str:
    """Provide stable serialization for non-JSON-native profile values."""
    return str(value)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a potentially messy Sarvam response."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Sarvam returned an empty borrower understanding response.")

    cleaned = text.strip()
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"No JSON object found in borrower understanding response: {text[:300]}"
        )

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


def _coerce_bool(value: Any) -> bool:
    """Coerce common model outputs to a real bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "haan", "ha"}
    return False


def _coerce_confidence(value: Any) -> float:
    """Normalize confidence to the required 0.0-1.0 range."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    return max(0.0, min(1.0, confidence))


def _history_text(
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
) -> str:
    """Flatten conversation context for local guardrails."""
    if isinstance(conversation_history, str):
        history = conversation_history
    else:
        history = json.dumps(
            conversation_history,
            ensure_ascii=False,
            default=_json_default,
        )
    return f"{history}\n{latest_borrower_message}".lower()


def _has_any(context: str, terms: tuple[str, ...]) -> bool:
    """Return whether any term appears in the flattened context."""
    return any(term in context for term in terms)


def _mentions_job_loss(context: str) -> bool:
    """Return whether the borrower explicitly described job loss."""
    return _has_any(
        context,
        (
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
            "naukri gayi",
            "kaam chala gaya",
            "काम चला गया",
        ),
    )


def _mentions_dispute_or_fraud(context: str) -> bool:
    """Return whether the borrower is disputing the loan or alleging fraud."""
    return _has_any(
        context,
        (
            "wrong loan",
            "not my loan",
            "fraud",
            "scam",
            "identity theft",
            "i did not take",
            "didn't take",
            "never took",
            "mera loan nahi",
            "mera nahi hai",
            "गलत लोन",
            "मेरा लोन नहीं",
            "धोखा",
        ),
    )


def _mentions_callback(context: str) -> bool:
    """Return whether the borrower requested a callback."""
    return _has_any(
        context,
        (
            "call back",
            "callback",
            "call me later",
            "call later",
            "baad mein call",
            "bad me call",
            "later call",
            "कल कॉल",
            "बाद में कॉल",
        ),
    )


def _mentions_human_escalation(context: str) -> bool:
    """Return whether the borrower requested a human or manager."""
    return _has_any(
        context,
        (
            "manager",
            "supervisor",
            "senior",
            "human",
            "real person",
            "agent se baat",
            "manager se baat",
            "किसी आदमी",
            "मैनेजर",
        ),
    )


def _mentions_partial_not_possible(context: str) -> bool:
    """Return whether the borrower rejected partial payment."""
    return _has_any(
        context,
        (
            "partial payment not possible",
            "cannot pay partial",
            "can't pay partial",
            "no partial",
            "part payment nahi",
            "partial nahi",
            "thoda bhi nahi",
            "कुछ भी नहीं",
        ),
    )


def _mentions_cannot_pay_today_but_later(context: str) -> bool:
    """Return whether borrower says today is impossible but later may work."""
    today_blocked = _has_any(
        context,
        (
            "cannot pay today",
            "can't pay today",
            "not today",
            "today not possible",
            "aaj nahi",
            "aaj possible nahi",
            "आज नहीं",
        ),
    )
    later_possible = _has_any(
        context,
        (
            "later",
            "tomorrow",
            "next week",
            "month end",
            "month-end",
            "salary",
            "payday",
            "kal",
            "parso",
            "agle hafte",
            "mahine ke end",
            "कल",
            "परसों",
        ),
    )
    return today_blocked and later_possible


def _profile_name(borrower_profile: dict[str, Any]) -> str:
    """Extract a borrower name from common profile fields."""
    for key in ("borrower_name", "name", "customer_name", "full_name"):
        name = _normalize_text(borrower_profile.get(key))
        if name:
            return name
    first_name = _normalize_text(borrower_profile.get("first_name"))
    last_name = _normalize_text(borrower_profile.get("last_name"))
    return " ".join(part for part in (first_name, last_name) if part).strip()


def _contains_yes_or_reminder(text: str) -> bool:
    """Return whether the latest borrower message confirms a reminder/date."""
    lowered = text.lower()
    return _has_any(
        lowered,
        (
            "yes",
            "haan",
            "ha",
            "okay",
            "ok",
            "reminder",
            "theek hai",
            "thik hai",
            "ठीक है",
            "हाँ",
        ),
    )


def _contains_monday_or_date(context: str) -> bool:
    """Return whether prior conversation contains a Monday/date confirmation."""
    return _has_any(
        context,
        (
            "monday",
            "date",
            "payment date",
            "promise date",
            "next monday",
            "सोमवार",
        ),
    )


def _normalize_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a borrower state payload."""
    intent = _normalize_text(payload.get("intent")).lower()
    if intent not in ALLOWED_INTENTS:
        raise ValueError(f"Invalid borrower intent: {intent!r}")

    job_status = _normalize_text(payload.get("job_status")).lower()
    if job_status not in ALLOWED_JOB_STATUSES:
        raise ValueError(f"Invalid borrower job_status: {job_status!r}")

    evidence = _normalize_text(payload.get("evidence"))
    if not evidence:
        raise ValueError("Borrower understanding state is missing evidence.")

    return {
        "borrower_name": _normalize_text(payload.get("borrower_name")),
        "intent": intent,
        "today_payment": _coerce_bool(payload.get("today_payment")),
        "promise_to_pay": _coerce_bool(payload.get("promise_to_pay")),
        "payment_date": _normalize_text(payload.get("payment_date")),
        "partial_payment_possible": _coerce_bool(
            payload.get("partial_payment_possible")
        ),
        "partial_payment_amount": _normalize_text(payload.get("partial_payment_amount")),
        "job_status": job_status,
        "income_date": _normalize_text(payload.get("income_date")),
        "financial_hardship": _coerce_bool(payload.get("financial_hardship")),
        "callback_requested": _coerce_bool(payload.get("callback_requested")),
        "human_escalation_requested": _coerce_bool(
            payload.get("human_escalation_requested")
        ),
        "dispute_or_fraud": _coerce_bool(payload.get("dispute_or_fraud")),
        "confidence": _coerce_confidence(payload.get("confidence")),
        "evidence": evidence,
    }


def _apply_borrower_state_guardrails(
    state: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
    borrower_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Enforce deterministic borrower-state constraints from explicit context."""
    context = _history_text(conversation_history, latest_borrower_message)
    latest_lower = latest_borrower_message.lower()
    state = dict(state)
    borrower_profile = borrower_profile or {}

    if not state.get("borrower_name"):
        profile_name = _profile_name(borrower_profile)
        if profile_name:
            state["borrower_name"] = profile_name

    if "next monday" in context:
        state["intent"] = "promise_to_pay"
        state["promise_to_pay"] = True
        state["payment_date"] = "next Monday"
        if state["evidence"] == FALLBACK_BORROWER_STATE["evidence"]:
            state["evidence"] = 'Borrower mentioned "next Monday" as payment timing.'

    if _contains_yes_or_reminder(latest_lower) and _contains_monday_or_date(context):
        state["intent"] = "promise_to_pay"
        state["promise_to_pay"] = True
        if not state["payment_date"]:
            state["payment_date"] = "next Monday" if "monday" in context else ""
        if state["evidence"] == FALLBACK_BORROWER_STATE["evidence"]:
            state["evidence"] = "Borrower confirmed reminder/date with yes or okay."

    if _mentions_job_loss(context):
        state["job_status"] = "lost_job"
        state["income_date"] = ""
        if state["intent"] == "unknown":
            state["intent"] = "hardship"
        state["financial_hardship"] = True

    if _mentions_cannot_pay_today_but_later(context):
        state["today_payment"] = False
        if not _mentions_human_escalation(context):
            state["human_escalation_requested"] = False

    if state["payment_date"]:
        state["promise_to_pay"] = True
        if state["intent"] in {"unknown", "cannot_pay"}:
            state["intent"] = "promise_to_pay"

    if _mentions_partial_not_possible(context):
        state["partial_payment_possible"] = False
        state["partial_payment_amount"] = ""

    if _mentions_callback(context):
        state["callback_requested"] = True
        if state["intent"] == "unknown":
            state["intent"] = "callback_request"

    if _mentions_human_escalation(context):
        state["human_escalation_requested"] = True

    if _mentions_dispute_or_fraud(context):
        state["dispute_or_fraud"] = True
        state["intent"] = "dispute"
        if state["evidence"] == FALLBACK_BORROWER_STATE["evidence"]:
            state["evidence"] = "Borrower said wrong loan or not my loan."

    if state["job_status"] == "lost_job":
        state["income_date"] = ""

    return state


def _build_messages(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
) -> list[dict[str, str]]:
    """Create the Sarvam prompt for one borrower-understanding turn."""
    schema = {
        "borrower_name": "",
        "intent": (
            "can_pay|cannot_pay|partial_payment|promise_to_pay|hardship|"
            "dispute|callback_request|unknown"
        ),
        "today_payment": False,
        "promise_to_pay": False,
        "payment_date": "",
        "partial_payment_possible": False,
        "partial_payment_amount": "",
        "job_status": "employed|lost_job|unknown",
        "income_date": "",
        "financial_hardship": False,
        "callback_requested": False,
        "human_escalation_requested": False,
        "dispute_or_fraud": False,
        "confidence": 0.0,
        "evidence": "",
    }

    return [
        {"role": "system", "content": BORROWER_UNDERSTANDING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Borrower profile:\n"
                + json.dumps(borrower_profile, ensure_ascii=False, default=_json_default)
                + "\n\nConversation history:\n"
                + json.dumps(conversation_history, ensure_ascii=False, default=_json_default)
                + "\n\nLatest borrower message:\n"
                + latest_borrower_message
                + "\n\nExtract the borrower state using this exact JSON schema:\n"
                + json.dumps(schema, ensure_ascii=False, indent=2, default=_json_default)
                + "\n\nRules:\n"
                '- If borrower commits to any future payment date/day/month-end/salary date, set intent="promise_to_pay", promise_to_pay=true, and payment_date.\n'
                '- If the borrower clearly commits to paying on a specific future date, for example tomorrow, Monday, next Monday, 5th July, or salary day, set intent="promise_to_pay", promise_to_pay=true, payment_date=the stated date, and financial_hardship=false.\n'
                '- If borrower says "next Monday", payment_date="next Monday".\n'
                "- If borrower says yes to reminder after date confirmation, keep promise_to_pay=true.\n"
                "- If borrower says cannot pay today but can pay later, today_payment=false but do not mark hardship unless hardship is stated.\n"
                "- Do not classify a normal promise-to-pay as hardship unless the borrower explicitly says they lost their job, have a medical emergency, severe financial hardship, or cannot commit to any payment timeline.\n"
                '- If borrower says job is gone/lost/no job, job_status="lost_job" and financial_hardship=true.\n'
                "- If borrower says partial payment is not possible, partial_payment_possible=false.\n"
                "- If borrower says wrong loan/fraud/not my loan, dispute_or_fraud=true.\n"
                "- If borrower asks for human/manager/callback, callback_requested=true.\n"
                "- Evidence must cite the borrower statement briefly."
            ),
        },
    ]


def _repair_messages(raw_output: str) -> list[dict[str, str]]:
    """Create a one-shot JSON repair prompt for malformed Sarvam output."""
    required_schema = {
        "borrower_name": "",
        "intent": (
            "can_pay|cannot_pay|partial_payment|promise_to_pay|hardship|"
            "dispute|callback_request|unknown"
        ),
        "today_payment": False,
        "promise_to_pay": False,
        "payment_date": "",
        "partial_payment_possible": False,
        "partial_payment_amount": "",
        "job_status": "employed|lost_job|unknown",
        "income_date": "",
        "financial_hardship": False,
        "callback_requested": False,
        "human_escalation_requested": False,
        "dispute_or_fraud": False,
        "confidence": 0.0,
        "evidence": "short explanation of what in the conversation supports this state",
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


def _fallback_state() -> dict[str, Any]:
    """Return the safe fallback borrower state."""
    return dict(FALLBACK_BORROWER_STATE)


def _parse_or_repair_borrower_state(raw: str) -> dict[str, Any]:
    """Parse borrower state, retrying Sarvam once for JSON repair if needed."""
    try:
        return _normalize_state(_extract_json_object(raw))
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        repaired = sarvam_reasoning(
            _repair_messages(raw),
            temperature=0.0,
            max_tokens=420,
        )
        return _normalize_state(_extract_json_object(repaired))
    except (SarvamLLMError, json.JSONDecodeError, ValueError):
        return _fallback_state()


def understand_borrower_state(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
) -> dict[str, Any]:
    """Ask Sarvam to produce a strict structured borrower state.

    Args:
        borrower_profile: Borrower/account metadata available to the workflow.
        conversation_history: Full prior conversation turns.
        latest_borrower_message: The newest borrower utterance.

    Returns:
        A validated borrower-state dictionary matching the required schema.

    Raises:
        SarvamLLMError: If Sarvam cannot be reached or returns no content.
        ValueError: If Sarvam returns malformed or invalid borrower-state JSON.
    """
    raw = sarvam_reasoning(
        _build_messages(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_message=latest_borrower_message,
        ),
        max_tokens=420,
    )
    state = _parse_or_repair_borrower_state(raw)
    return _apply_borrower_state_guardrails(
        state=state,
        conversation_history=conversation_history,
        latest_borrower_message=latest_borrower_message,
        borrower_profile=borrower_profile,
    )


def understand_borrower_state_json(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
) -> str:
    """Return a compact JSON string for the structured borrower state."""
    return json.dumps(
        understand_borrower_state(
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
    """Read borrower-understanding inputs from ADK state and user content."""
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
    """Build an ADK Event containing the borrower-state JSON response."""
    if not ADK_AVAILABLE:
        return text

    return Event(
        author=author,
        invocation_id=getattr(parent_context, "invocation_id", ""),
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        actions=EventActions(),
        timestamp=time.time(),
    )


class BorrowerUnderstandingAgent(BaseAgent):
    """ADK specialist agent that structures borrower conversation state."""

    async def _run_async_impl(self, parent_context: Any):
        """Run one borrower-understanding turn and yield a JSON-only event."""
        borrower_profile, conversation_history, latest_message = _context_payload(
            parent_context
        )
        state = understand_borrower_state_json(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_message=latest_message,
        )
        yield _adk_event(parent_context, self.name, state)

    async def _run_live_impl(self, parent_context: Any):
        """Handle live invocations through the same JSON-only path."""
        async for event in self._run_async_impl(parent_context):
            yield event


root_agent = BorrowerUnderstandingAgent(
    name="borrower_understanding_agent",
    description=(
        "Extracts structured borrower state from messy borrower conversation "
        "and returns strict JSON only."
    ),
)


__all__ = [
    "BorrowerUnderstandingAgent",
    "understand_borrower_state",
    "understand_borrower_state_json",
    "root_agent",
]

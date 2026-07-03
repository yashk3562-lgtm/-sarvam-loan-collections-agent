"""Sarvam-powered Google ADK compliance specialist for collections.

This agent checks whether a collections conversation is compliant. It does not
negotiate, route, update CRM, create reminders, or produce borrower-facing
speech.
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


ALLOWED_COMPLIANCE_STATUSES = frozenset({"pass", "warning", "fail"})
ALLOWED_RISK_LEVELS = frozenset({"low", "medium", "high"})
ALLOWED_RECOMMENDED_ACTIONS = frozenset(
    {"continue", "human_review", "stop_contact", "escalate"}
)
RISK_ORDER = {"low": 0, "medium": 1, "high": 2}

COMPLIANCE_SYSTEM_PROMPT = """You are the Compliance Specialist Agent for a loan collections workflow.

Scope:
- Check whether the collections conversation is compliant.
- Do not negotiate.
- Do not route.
- Do not update CRM.
- Do not create reminders.
- Do not produce borrower-facing speech.

Input:
- borrower profile
- full conversation history
- latest borrower message
- borrower structured state if available

Check for:
- threats
- harassment
- legal pressure
- shame language
- repeated pressure after refusal
- wrong borrower / fraud claim
- sensitive hardship
- human callback request
- abusive borrower
- agent saying anything non-compliant

Rules:
- If wrong borrower, fraud, or dispute is detected, recommended_action must be "human_review".
- If agent used threat, shame, or legal pressure, compliance_status must be "fail".
- If borrower has hardship, risk_level should be at least "medium" but hardship alone is not a compliance violation.
- Do not over-escalate when the borrower simply cannot pay today.

Strict output schema:
{
  "compliance_status": "pass|warning|fail",
  "risk_level": "low|medium|high",
  "violations": [],
  "reason": "",
  "recommended_action": "continue|human_review|stop_contact|escalate"
}

Output rules:
- Return ONLY one valid JSON object.
- Do not explain.
- Do not include markdown.
- Do not include reasoning.
- Do not include numbered analysis.
- The first character must be { and the last character must be }.
"""

FALLBACK_COMPLIANCE_DECISION = {
    "compliance_status": "warning",
    "risk_level": "medium",
    "violations": [],
    "reason": (
        "Fallback used because Sarvam did not return clean compliance JSON. "
        "No explicit violation was confirmed."
    ),
    "recommended_action": "human_review",
}

CLEAN_PROMISE_FALLBACK_COMPLIANCE_DECISION = {
    "compliance_status": "pass",
    "risk_level": "low",
    "violations": [],
    "reason": (
        "Fallback used because Sarvam did not return clean compliance JSON. "
        "Conversation appears to be a standard promise-to-pay flow with no "
        "explicit compliance violation."
    ),
    "recommended_action": "continue",
}


def _json_default(value: Any) -> str:
    """Provide stable serialization for non-JSON-native profile values."""
    return str(value)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a potentially messy Sarvam response."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Sarvam returned an empty compliance response.")

    cleaned = text.strip()
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in compliance response: {text[:300]}")

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


def _normalize_violations(value: Any) -> list[str]:
    """Normalize violations to a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value] if value.strip() else []
    if not isinstance(value, list):
        raise ValueError("Compliance violations must be a list.")
    return [_normalize_text(item) for item in value if _normalize_text(item)]


def _min_risk(risk_level: str, minimum: str) -> str:
    """Raise risk level to a minimum threshold."""
    if RISK_ORDER[risk_level] < RISK_ORDER[minimum]:
        return minimum
    return risk_level


def _history_text(
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
    borrower_state: dict[str, Any] | None,
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
    state = json.dumps(borrower_state or {}, ensure_ascii=False, default=_json_default)
    return f"{history}\n{latest_borrower_message}\n{state}".lower()


def _agent_history_text(conversation_history: list[dict[str, Any]] | list[str] | str) -> str:
    """Flatten likely agent messages for agent-side compliance checks."""
    if isinstance(conversation_history, str):
        return conversation_history.lower()
    agent_messages: list[str] = []
    for item in conversation_history:
        if isinstance(item, str):
            agent_messages.append(item)
            continue
        role = str(item.get("role") or item.get("author") or "").lower()
        if role in {"agent", "assistant", "model", "collector", "collection_agent"}:
            content = item.get("content") or item.get("message") or item.get("text") or ""
            agent_messages.append(str(content))
    return "\n".join(agent_messages).lower()


def _has_any(context: str, terms: tuple[str, ...]) -> bool:
    """Return whether any term appears in flattened text."""
    return any(term in context for term in terms)


def _append_violation(violations: list[str], violation: str) -> list[str]:
    """Append a violation if it is not already present."""
    if violation not in violations:
        violations.append(violation)
    return violations


def _mentions_dispute_or_fraud(context: str) -> bool:
    """Return whether borrower disputes the loan or claims fraud/wrong party."""
    return _has_any(
        context,
        (
            "wrong borrower",
            "wrong loan",
            "not my loan",
            "fraud",
            "scam",
            "identity theft",
            "i did not take",
            "didn't take",
            "never took",
            "dispute",
            "mera loan nahi",
            "mera nahi hai",
            "गलत लोन",
            "मेरा लोन नहीं",
            "धोखा",
        ),
    )


def _mentions_hardship(context: str, borrower_state: dict[str, Any] | None) -> bool:
    """Return whether hardship is explicit in context or structured state."""
    if borrower_state:
        if borrower_state.get("financial_hardship") is True:
            return True
        if str(borrower_state.get("intent", "")).lower() == "hardship":
            return True
        if str(borrower_state.get("job_status", "")).lower() == "lost_job":
            return True
    return _has_any(
        context,
        (
            "lost my job",
            "lost job",
            "job loss",
            "unemployed",
            "medical",
            "hospital",
            "accident",
            "death in family",
            "family emergency",
            "salary delayed",
            "financial hardship",
            "naukri chali",
            "नौकरी चली",
            "बीमारी",
            "अस्पताल",
        ),
    )


def _mentions_human_or_callback(context: str, borrower_state: dict[str, Any] | None) -> bool:
    """Return whether borrower requested human review or callback."""
    if borrower_state and (
        borrower_state.get("callback_requested") is True
        or borrower_state.get("human_escalation_requested") is True
    ):
        return True
    return _has_any(
        context,
        (
            "call back",
            "callback",
            "call me later",
            "manager",
            "supervisor",
            "senior",
            "human",
            "real person",
            "manager se baat",
            "baad mein call",
            "मैनेजर",
        ),
    )


def _agent_used_threat_shame_or_legal_pressure(agent_context: str) -> bool:
    """Detect explicit agent-side threats, shame, or legal-pressure language."""
    return _has_any(
        agent_context,
        (
            "we will arrest",
            "you will be arrested",
            "police will come",
            "legal action immediately",
            "court case today",
            "jail",
            "seize your property",
            "tell your family",
            "tell your employer",
            "shame",
            "shameless",
            "defaulter in front of",
            "bad person",
            "fraudster",
            "गिरफ्तार",
            "जेल",
            "शर्म",
        ),
    )


def _mentions_abusive_borrower(context: str) -> bool:
    """Detect borrower abuse that raises operational risk."""
    return _has_any(
        context,
        (
            "abuse",
            "abusive",
            "idiot",
            "stupid",
            "shut up",
            "gaali",
            "गाली",
        ),
    )


def _normalize_decision(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a compliance decision."""
    compliance_status = _normalize_text(payload.get("compliance_status")).lower()
    if compliance_status not in ALLOWED_COMPLIANCE_STATUSES:
        raise ValueError(f"Invalid compliance_status: {compliance_status!r}")

    risk_level = _normalize_text(payload.get("risk_level")).lower()
    if risk_level not in ALLOWED_RISK_LEVELS:
        raise ValueError(f"Invalid compliance risk_level: {risk_level!r}")

    recommended_action = _normalize_text(payload.get("recommended_action")).lower()
    if recommended_action not in ALLOWED_RECOMMENDED_ACTIONS:
        raise ValueError(f"Invalid compliance recommended_action: {recommended_action!r}")

    reason = _normalize_text(payload.get("reason"))
    if not reason:
        raise ValueError("Compliance decision is missing a reason.")

    return {
        "compliance_status": compliance_status,
        "risk_level": risk_level,
        "violations": _normalize_violations(payload.get("violations")),
        "reason": reason,
        "recommended_action": recommended_action,
    }


def _is_clean_promise_to_pay_state(borrower_state: dict[str, Any] | None) -> bool:
    """Return whether fallback can safely pass a clean promise-to-pay flow."""
    state = borrower_state or {}
    return (
        state.get("promise_to_pay") is True
        and state.get("dispute_or_fraud") is False
        and state.get("human_escalation_requested") is False
        and state.get("financial_hardship") is False
    )


def _fallback_decision(borrower_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the safe compliance fallback decision."""
    if _is_clean_promise_to_pay_state(borrower_state):
        return dict(CLEAN_PROMISE_FALLBACK_COMPLIANCE_DECISION)
    return dict(FALLBACK_COMPLIANCE_DECISION)


def _parse_or_fallback_compliance_decision(
    raw: str,
    borrower_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse compliance JSON, returning fallback instead of raising."""
    try:
        return _normalize_decision(_extract_json_object(raw))
    except (json.JSONDecodeError, ValueError):
        return _fallback_decision(borrower_state)


def _apply_compliance_guardrails(
    decision: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
    borrower_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Enforce deterministic compliance outcomes for explicit signals."""
    context = _history_text(conversation_history, latest_borrower_message, borrower_state)
    agent_context = _agent_history_text(conversation_history)
    decision = dict(decision)
    violations = list(decision["violations"])

    if _mentions_dispute_or_fraud(context):
        decision["recommended_action"] = "human_review"
        decision["risk_level"] = _min_risk(decision["risk_level"], "medium")
        if decision["compliance_status"] == "pass":
            decision["compliance_status"] = "warning"
        _append_violation(violations, "wrong_borrower_or_fraud_dispute")

    if _agent_used_threat_shame_or_legal_pressure(agent_context):
        decision["compliance_status"] = "fail"
        decision["risk_level"] = "high"
        decision["recommended_action"] = "human_review"
        _append_violation(violations, "agent_threat_shame_or_legal_pressure")

    if _mentions_hardship(context, borrower_state):
        decision["risk_level"] = _min_risk(decision["risk_level"], "medium")
        if not violations and decision["recommended_action"] != "human_review":
            decision["compliance_status"] = "pass"
            decision["recommended_action"] = "continue"

    if _mentions_human_or_callback(context, borrower_state):
        decision["risk_level"] = _min_risk(decision["risk_level"], "medium")
        if decision["recommended_action"] == "continue":
            decision["recommended_action"] = "human_review"
        if decision["compliance_status"] == "pass":
            decision["compliance_status"] = "warning"
        _append_violation(violations, "human_or_callback_request")

    if _mentions_abusive_borrower(context):
        decision["risk_level"] = _min_risk(decision["risk_level"], "medium")
        if decision["compliance_status"] == "pass":
            decision["compliance_status"] = "warning"
        _append_violation(violations, "abusive_borrower")

    decision["violations"] = violations
    return decision


def _build_messages(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
    borrower_state: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Create the Sarvam prompt for one compliance check."""
    compliance_input = {
        "borrower_profile": borrower_profile,
        "conversation_history": conversation_history,
        "latest_borrower_message": latest_borrower_message,
        "borrower_state": borrower_state or {},
    }

    return [
        {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Check this collections conversation for compliance. "
                "Return ONLY one valid JSON object. Do not explain. Do not "
                "include markdown, reasoning, or numbered analysis. The first "
                "character must be { and the last character must be }.\n\n"
                + json.dumps(compliance_input, ensure_ascii=False, default=_json_default)
            ),
        },
    ]


def check_compliance(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
    borrower_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ask Sarvam to produce a strict compliance decision.

    Args:
        borrower_profile: Borrower/account metadata available to the workflow.
        conversation_history: Full prior conversation turns.
        latest_borrower_message: The newest borrower utterance.
        borrower_state: Optional structured borrower state.

    Returns:
        A validated compliance decision matching the required schema.

    Raises:
        SarvamLLMError: If Sarvam cannot be reached or returns no content.
        ValueError: If Sarvam returns malformed or invalid compliance JSON.
    """
    raw = sarvam_reasoning(
        _build_messages(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_message=latest_borrower_message,
            borrower_state=borrower_state,
        ),
        max_tokens=360,
    )
    decision = _parse_or_fallback_compliance_decision(raw, borrower_state)
    return _apply_compliance_guardrails(
        decision=decision,
        conversation_history=conversation_history,
        latest_borrower_message=latest_borrower_message,
        borrower_state=borrower_state,
    )


def check_compliance_json(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
    borrower_state: dict[str, Any] | None = None,
) -> str:
    """Return a compact JSON string for the compliance decision."""
    return json.dumps(
        check_compliance(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_message=latest_borrower_message,
            borrower_state=borrower_state,
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


def _context_payload(parent_context: Any) -> tuple[dict[str, Any], Any, str, dict[str, Any] | None]:
    """Read compliance inputs from ADK session state and user content."""
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

    return borrower_profile, conversation_history, latest_borrower_message, borrower_state


def _adk_event(parent_context: Any, author: str, text: str) -> Any:
    """Build an ADK Event containing the compliance JSON response."""
    if not ADK_AVAILABLE:
        return text

    return Event(
        author=author,
        invocation_id=getattr(parent_context, "invocation_id", ""),
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        actions=EventActions(),
        timestamp=time.time(),
    )


class ComplianceAgent(BaseAgent):
    """ADK specialist agent that checks collections compliance only."""

    async def _run_async_impl(self, parent_context: Any):
        """Run one compliance check and yield a JSON-only ADK event."""
        borrower_profile, conversation_history, latest_message, borrower_state = (
            _context_payload(parent_context)
        )
        decision = check_compliance_json(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_message=latest_message,
            borrower_state=borrower_state,
        )
        yield _adk_event(parent_context, self.name, decision)

    async def _run_live_impl(self, parent_context: Any):
        """Handle live invocations through the same compliance JSON path."""
        async for event in self._run_async_impl(parent_context):
            yield event


root_agent = ComplianceAgent(
    name="compliance_agent",
    description=(
        "Checks collections conversation compliance and returns strict JSON "
        "with pass, warning, or fail status."
    ),
)


__all__ = [
    "ComplianceAgent",
    "check_compliance",
    "check_compliance_json",
    "root_agent",
]

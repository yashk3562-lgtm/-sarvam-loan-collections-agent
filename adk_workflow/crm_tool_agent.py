"""Google ADK CRM tool executor for collections workflow outcomes.

This agent does not call Sarvam. It deterministically executes CRM-side tools
from structured workflow decisions and returns strict JSON describing what ran.
"""

from __future__ import annotations

import json
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


FINAL_STATUSES = frozenset(
    {"continue", "promise_to_pay", "escalated", "restructure_requested"}
)


def _json_default(value: Any) -> str:
    """Provide stable serialization for non-JSON-native values."""
    return str(value)


def _normalize_text(value: Any) -> str:
    """Normalize free-form input text."""
    if value is None:
        return ""
    return str(value).strip()


def _tool_result(action: str, borrower_profile: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic tool execution receipt."""
    borrower_id = (
        borrower_profile.get("borrower_id")
        or borrower_profile.get("customer_id")
        or borrower_profile.get("account_id")
        or borrower_profile.get("loan_id")
        or ""
    )
    return {
        "action": action,
        "borrower_id": _normalize_text(borrower_id),
        "status": "success",
        "payload": payload,
    }


def update_crm_record(
    borrower_profile: dict[str, Any],
    borrower_state: dict[str, Any],
    final_status: str,
) -> dict[str, Any]:
    """Tool: update the CRM record with the final workflow outcome."""
    return _tool_result(
        "update_crm_record",
        borrower_profile,
        {
            "final_status": final_status,
            "borrower_state": borrower_state,
        },
    )


def create_payment_reminder(
    borrower_profile: dict[str, Any],
    borrower_state: dict[str, Any],
) -> dict[str, Any]:
    """Tool: create a payment reminder for a promise-to-pay outcome."""
    return _tool_result(
        "create_payment_reminder",
        borrower_profile,
        {
            "payment_date": _normalize_text(borrower_state.get("payment_date")),
            "partial_payment_amount": _normalize_text(
                borrower_state.get("partial_payment_amount")
            ),
        },
    )


def create_human_escalation(
    borrower_profile: dict[str, Any],
    borrower_state: dict[str, Any],
    specialist_decisions: Any,
) -> dict[str, Any]:
    """Tool: create a human escalation task for the account."""
    return _tool_result(
        "create_human_escalation",
        borrower_profile,
        {
            "borrower_state": borrower_state,
            "specialist_decisions": specialist_decisions,
        },
    )


def create_restructure_request(
    borrower_profile: dict[str, Any],
    borrower_state: dict[str, Any],
    specialist_decisions: Any,
) -> dict[str, Any]:
    """Tool: create a restructure request for the account."""
    return _tool_result(
        "create_restructure_request",
        borrower_profile,
        {
            "borrower_state": borrower_state,
            "specialist_decisions": specialist_decisions,
        },
    )


def _walk_values(value: Any) -> list[str]:
    """Flatten nested decision values into lowercase strings for inference."""
    if isinstance(value, dict):
        flattened: list[str] = []
        for item in value.values():
            flattened.extend(_walk_values(item))
        return flattened
    if isinstance(value, list):
        flattened = []
        for item in value:
            flattened.extend(_walk_values(item))
        return flattened
    return [_normalize_text(value).lower()] if value is not None else []


def _infer_final_status(
    borrower_state: dict[str, Any],
    specialist_decisions: Any,
    final_outcome: Any,
) -> str:
    """Normalize final outcome into the small CRM action vocabulary."""
    values = [value.replace("-", "_") for value in _walk_values(final_outcome)]
    if not values:
        values = [value.replace("-", "_") for value in _walk_values(specialist_decisions)]

    joined = " ".join(values)
    if "continue" in joined and not any(
        token in joined for token in ("promise", "escalat", "human_review", "restructure")
    ):
        return "continue"
    if "restructure" in joined:
        return "restructure_requested"
    if "promise_to_pay" in joined or "promise to pay" in joined:
        return "promise_to_pay"
    if "escalat" in joined or "human_review" in joined or "human review" in joined:
        return "escalated"

    if borrower_state.get("promise_to_pay") is True or borrower_state.get("payment_date"):
        return "promise_to_pay"

    return "continue"


def _append_action(
    actions: list[str],
    receipts: list[dict[str, Any]],
    seen: set[str],
    name: str,
    receipt: dict[str, Any],
) -> None:
    """Append one action and receipt, skipping duplicates."""
    if name in seen:
        return
    seen.add(name)
    actions.append(name)
    receipts.append(receipt)


def _validate_result(result: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the CRM tool result schema."""
    final_status = _normalize_text(result.get("final_status"))
    if final_status not in FINAL_STATUSES:
        raise ValueError(f"Invalid CRM final_status: {final_status!r}")

    actions = result.get("actions")
    if not isinstance(actions, list):
        raise ValueError("CRM result actions must be a list.")

    return {
        "crm_updated": bool(result.get("crm_updated")),
        "reminder_created": bool(result.get("reminder_created")),
        "escalation_created": bool(result.get("escalation_created")),
        "restructure_created": bool(result.get("restructure_created")),
        "actions": [_normalize_text(action) for action in actions if _normalize_text(action)],
        "final_status": final_status,
    }


def run_crm_tools(
    borrower_profile: dict[str, Any],
    borrower_state: dict[str, Any],
    specialist_decisions: Any,
    final_outcome: Any,
) -> dict[str, Any]:
    """Execute CRM tools for a final workflow outcome.

    Rules:
    - Promise-to-Pay creates CRM update and payment reminder.
    - Escalation creates CRM update and human escalation.
    - Restructure Requested creates CRM update and restructure request.
    - Continue creates no final CRM event.
    - Duplicate actions are skipped.
    """
    borrower_profile = dict(borrower_profile or {})
    borrower_state = dict(borrower_state or {})
    final_status = _infer_final_status(
        borrower_state=borrower_state,
        specialist_decisions=specialist_decisions,
        final_outcome=final_outcome,
    )

    actions: list[str] = []
    receipts: list[dict[str, Any]] = []
    seen: set[str] = set()

    if final_status == "promise_to_pay":
        _append_action(
            actions,
            receipts,
            seen,
            "update_crm_record",
            update_crm_record(borrower_profile, borrower_state, final_status),
        )
        _append_action(
            actions,
            receipts,
            seen,
            "create_payment_reminder",
            create_payment_reminder(borrower_profile, borrower_state),
        )
    elif final_status == "escalated":
        _append_action(
            actions,
            receipts,
            seen,
            "update_crm_record",
            update_crm_record(borrower_profile, borrower_state, final_status),
        )
        _append_action(
            actions,
            receipts,
            seen,
            "create_human_escalation",
            create_human_escalation(
                borrower_profile, borrower_state, specialist_decisions
            ),
        )
    elif final_status == "restructure_requested":
        _append_action(
            actions,
            receipts,
            seen,
            "update_crm_record",
            update_crm_record(borrower_profile, borrower_state, final_status),
        )
        _append_action(
            actions,
            receipts,
            seen,
            "create_restructure_request",
            create_restructure_request(
                borrower_profile, borrower_state, specialist_decisions
            ),
        )

    result = {
        "crm_updated": "update_crm_record" in seen,
        "reminder_created": "create_payment_reminder" in seen,
        "escalation_created": "create_human_escalation" in seen,
        "restructure_created": "create_restructure_request" in seen,
        "actions": actions,
        "final_status": final_status,
        "_tool_receipts": receipts,
    }
    validated = _validate_result(result)
    return validated


def run_crm_tools_json(
    borrower_profile: dict[str, Any],
    borrower_state: dict[str, Any],
    specialist_decisions: Any,
    final_outcome: Any,
) -> str:
    """Return a compact strict JSON string for CRM tool execution."""
    return json.dumps(
        run_crm_tools(
            borrower_profile=borrower_profile,
            borrower_state=borrower_state,
            specialist_decisions=specialist_decisions,
            final_outcome=final_outcome,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )


def _content_text(content: Any) -> str:
    """Extract text from an ADK/GenAI content object."""
    if content is None:
        return ""

    parts = getattr(content, "parts", None) or []
    texts = [str(getattr(part, "text", "") or "") for part in parts]
    text = "\n".join(part for part in texts if part).strip()
    return text or str(content)


def _context_payload(parent_context: Any) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """Read CRM tool inputs from ADK session state and user content."""
    session = getattr(parent_context, "session", None)
    state = getattr(session, "state", {}) or {}
    latest_text = _content_text(getattr(parent_context, "user_content", None))

    borrower_profile = dict(
        state.get("borrower_profile")
        or state.get("borrower")
        or state.get("account")
        or {}
    )
    borrower_state = dict(
        state.get("borrower_state") or state.get("structured_borrower_state") or {}
    )
    specialist_decisions = state.get("specialist_decisions") or state.get("decisions") or {}
    final_outcome = state.get("final_outcome") or state.get("final_status") or "continue"

    try:
        payload = json.loads(latest_text) if latest_text else {}
    except json.JSONDecodeError:
        payload = {}

    if isinstance(payload, dict):
        borrower_profile = dict(payload.get("borrower_profile") or borrower_profile)
        borrower_state = dict(
            payload.get("borrower_state")
            or payload.get("structured_borrower_state")
            or borrower_state
        )
        specialist_decisions = (
            payload.get("specialist_decisions")
            or payload.get("decisions")
            or specialist_decisions
        )
        final_outcome = (
            payload.get("final_outcome")
            or payload.get("final_status")
            or final_outcome
        )

    return borrower_profile, borrower_state, specialist_decisions, final_outcome


def _adk_event(parent_context: Any, author: str, text: str) -> Any:
    """Build an ADK Event containing the CRM tool JSON response."""
    if not ADK_AVAILABLE:
        return text

    return Event(
        author=author,
        invocation_id=getattr(parent_context, "invocation_id", ""),
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        actions=EventActions(),
        timestamp=time.time(),
    )


class CRMToolAgent(BaseAgent):
    """ADK agent that executes deterministic CRM tools for final outcomes."""

    async def _run_async_impl(self, parent_context: Any):
        """Run CRM tools once and yield a JSON-only ADK event."""
        borrower_profile, borrower_state, specialist_decisions, final_outcome = (
            _context_payload(parent_context)
        )
        result = run_crm_tools_json(
            borrower_profile=borrower_profile,
            borrower_state=borrower_state,
            specialist_decisions=specialist_decisions,
            final_outcome=final_outcome,
        )
        yield _adk_event(parent_context, self.name, result)

    async def _run_live_impl(self, parent_context: Any):
        """Handle live invocations through the same CRM tool path."""
        async for event in self._run_async_impl(parent_context):
            yield event


root_agent = CRMToolAgent(
    name="crm_tool_agent",
    description=(
        "Executes deterministic CRM tools from structured workflow decisions "
        "and returns strict JSON only."
    ),
)


__all__ = [
    "CRMToolAgent",
    "run_crm_tools",
    "run_crm_tools_json",
    "root_agent",
]

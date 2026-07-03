"""Post-call Google ADK workflow orchestrator for collections.

The workflow coordinates the existing specialist functions. It does not call
Gemini directly, and any Sarvam reasoning happens only through the specialist
agents' existing Sarvam helper paths.
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

from .borrower_understanding_agent import understand_borrower_state
from .compliance_agent import check_compliance
from .crm_tool_agent import run_crm_tools
from .hardship_agent import analyze_hardship
from .negotiation_agent import negotiate_payment
from .supervisor import decide_next_agent


WORKFLOW_KEYS = (
    "borrower_state",
    "supervisor_decision",
    "specialist_decision",
    "compliance_decision",
    "crm_tool_result",
    "workflow_summary",
)
FINAL_OUTCOMES = frozenset(
    {"promise_to_pay", "escalated", "escalate", "restructure_requested"}
)
PROMISE_TO_PAY_OUTCOMES = frozenset(
    {"promise_to_pay", "promise_date_captured"}
)


def _json_default(value: Any) -> str:
    """Provide stable serialization for non-JSON-native values."""
    return str(value)


def _normalize_text(value: Any) -> str:
    """Normalize free-form text values."""
    if value is None:
        return ""
    return str(value).strip()


def _normalized_outcome(value: Any) -> str:
    """Normalize final outcome labels for workflow decisions."""
    if isinstance(value, dict):
        for key in ("final_status", "status", "outcome", "recommended_action"):
            if value.get(key):
                return _normalized_outcome(value[key])
        return " ".join(_normalized_outcome(item) for item in value.values()).strip()
    if isinstance(value, list):
        return " ".join(_normalized_outcome(item) for item in value).strip()
    return _normalize_text(value).lower().replace("-", "_").replace(" ", "_")


def _is_final_outcome(final_outcome: Any) -> bool:
    """Return whether CRM tools should execute for this outcome."""
    normalized = _normalized_outcome(final_outcome)
    return any(outcome in normalized for outcome in FINAL_OUTCOMES) or any(
        outcome in normalized for outcome in PROMISE_TO_PAY_OUTCOMES
    )


def _normalized_crm_outcome(final_outcome: Any) -> str:
    """Normalize final outcome labels before CRM tool execution."""
    normalized = _normalized_outcome(final_outcome)
    if any(outcome in normalized for outcome in PROMISE_TO_PAY_OUTCOMES):
        return "promise_to_pay"
    if "escalat" in normalized:
        return "escalated"
    if "restructure" in normalized:
        return "restructure_requested"
    return normalized or "continue"


def _empty_crm_result(final_outcome: Any) -> dict[str, Any]:
    """Return the strict CRM result shape when no final CRM event is allowed."""
    final_status = _normalized_outcome(final_outcome) or "continue"
    if final_status not in FINAL_OUTCOMES:
        final_status = "continue"
    if final_status == "escalate":
        final_status = "escalated"
    return {
        "crm_updated": False,
        "reminder_created": False,
        "escalation_created": False,
        "restructure_created": False,
        "actions": [],
        "final_status": final_status,
    }


def _fallback_supervisor_decision(
    borrower_state: dict[str, Any],
    final_outcome: Any,
) -> dict[str, Any]:
    """Create a valid supervisor decision when routing output is unusable."""
    if borrower_state.get("financial_hardship") is True or _normalize_text(
        borrower_state.get("job_status")
    ).lower() == "lost_job":
        next_agent = "hardship"
    else:
        next_agent = "negotiation"

    return {
        "next_agent": next_agent,
        "reason": "Fallback routing used because supervisor returned non-JSON output.",
        "confidence": 0.65,
    }


def _specialist_decision(
    next_agent: str,
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
    borrower_state: dict[str, Any],
) -> dict[str, Any]:
    """Call the business specialist chosen by the supervisor.

    Compliance is excluded here because it runs mandatorily after the business
    specialist and before CRM tools.
    """
    if next_agent == "hardship":
        return analyze_hardship(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_utterance=latest_borrower_message,
        )
    if next_agent == "negotiation":
        return negotiate_payment(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_message=latest_borrower_message,
        )
    return {}


def _workflow_summary(
    supervisor_decision: dict[str, Any],
    compliance_decision: dict[str, Any],
    crm_tool_result: dict[str, Any],
) -> str:
    """Build a concise internal workflow summary."""
    next_agent = _normalize_text(supervisor_decision.get("next_agent")) or "unknown"
    compliance_status = (
        _normalize_text(compliance_decision.get("compliance_status")) or "unknown"
    )
    crm_status = _normalize_text(crm_tool_result.get("final_status")) or "continue"
    actions = crm_tool_result.get("actions") or []
    action_text = ", ".join(actions) if actions else "no CRM action"
    return (
        f"Borrower state extracted, supervisor selected {next_agent}, "
        f"compliance status is {compliance_status}, CRM final status is "
        f"{crm_status} with {action_text}."
    )


def _validate_workflow_result(result: dict[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the post-call workflow result schema."""
    missing = [key for key in WORKFLOW_KEYS if key not in result]
    if missing:
        raise ValueError(f"Workflow result missing keys: {missing}")

    for key in WORKFLOW_KEYS[:-1]:
        if not isinstance(result[key], dict):
            raise ValueError(f"Workflow result field {key!r} must be an object.")

    workflow_summary = _normalize_text(result.get("workflow_summary"))
    if not workflow_summary:
        raise ValueError("Workflow result is missing workflow_summary.")

    return {
        "borrower_state": result["borrower_state"],
        "supervisor_decision": result["supervisor_decision"],
        "specialist_decision": result["specialist_decision"],
        "compliance_decision": result["compliance_decision"],
        "crm_tool_result": result["crm_tool_result"],
        "workflow_summary": workflow_summary,
    }


def run_post_call_workflow(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
    final_outcome: Any,
) -> dict[str, Any]:
    """Run the post-call workflow and return a strict JSON-compatible object.

    Execution order:
    1. Create structured borrower state.
    2. Ask the supervisor which specialist should run next.
    3. Run hardship or negotiation business specialist when selected.
    4. Always run compliance before CRM tools.
    5. Run CRM tools only for final outcomes.

    Args:
        borrower_profile: Borrower/account metadata available to the workflow.
        conversation_history: Full prior conversation turns.
        latest_borrower_message: The newest borrower utterance.
        final_outcome: Workflow outcome used to decide whether CRM tools run.

    Returns:
        A strict JSON-compatible workflow result with borrower state,
        supervisor decision, specialist decision, compliance decision, CRM tool
        result, and summary.
    """
    borrower_profile = dict(borrower_profile or {})
    latest_borrower_message = _normalize_text(latest_borrower_message)

    borrower_state = understand_borrower_state(
        borrower_profile=borrower_profile,
        conversation_history=conversation_history,
        latest_borrower_message=latest_borrower_message,
    )
    is_promise_to_pay = borrower_state.get("promise_to_pay") is True
    has_hardship = borrower_state.get("financial_hardship") is True
    has_dispute = borrower_state.get("dispute_or_fraud") is True

    if is_promise_to_pay and not has_hardship and not has_dispute:
        supervisor_decision = {
            "next_agent": "negotiation",
            "reason": "Promise-to-pay detected.",
            "confidence": 0.95,
        }
    else:
        try:
            supervisor_decision = decide_next_agent(
                borrower_profile=borrower_profile,
                conversation_history=conversation_history,
                latest_borrower_message=latest_borrower_message,
            )
        except Exception:
            supervisor_decision = _fallback_supervisor_decision(
                borrower_state=borrower_state,
                final_outcome=final_outcome,
            )

    next_agent = _normalize_text(supervisor_decision.get("next_agent")).lower()
    specialist_decision = _specialist_decision(
        next_agent=next_agent,
        borrower_profile=borrower_profile,
        conversation_history=conversation_history,
        latest_borrower_message=latest_borrower_message,
        borrower_state=borrower_state,
    )

    compliance_decision = check_compliance(
        borrower_profile=borrower_profile,
        conversation_history=conversation_history,
        latest_borrower_message=latest_borrower_message,
        borrower_state=borrower_state,
    )

    specialist_decisions = {
        "supervisor": supervisor_decision,
        "specialist": specialist_decision,
        "compliance": compliance_decision,
    }
    if _is_final_outcome(final_outcome):
        crm_tool_result = run_crm_tools(
            borrower_profile=borrower_profile,
            borrower_state=borrower_state,
            specialist_decisions=specialist_decisions,
            final_outcome=_normalized_crm_outcome(final_outcome),
        )
    else:
        crm_tool_result = _empty_crm_result(final_outcome)

    result = {
        "borrower_state": borrower_state,
        "supervisor_decision": supervisor_decision,
        "specialist_decision": specialist_decision,
        "compliance_decision": compliance_decision,
        "crm_tool_result": crm_tool_result,
        "workflow_summary": _workflow_summary(
            supervisor_decision=supervisor_decision,
            compliance_decision=compliance_decision,
            crm_tool_result=crm_tool_result,
        ),
    }
    return _validate_workflow_result(result)


def run_post_call_workflow_json(
    borrower_profile: dict[str, Any],
    conversation_history: list[dict[str, Any]] | list[str] | str,
    latest_borrower_message: str,
    final_outcome: Any,
) -> str:
    """Run the post-call workflow and return compact strict JSON."""
    return json.dumps(
        run_post_call_workflow(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_message=latest_borrower_message,
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


def _context_payload(parent_context: Any) -> tuple[dict[str, Any], Any, str, Any]:
    """Read post-call workflow inputs from ADK session state and user content."""
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
    final_outcome = state.get("final_outcome") or state.get("final_status") or "continue"

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
        final_outcome = (
            payload.get("final_outcome")
            or payload.get("final_status")
            or final_outcome
        )

    return borrower_profile, conversation_history, latest_borrower_message, final_outcome


def _adk_event(parent_context: Any, author: str, text: str) -> Any:
    """Build an ADK Event containing the workflow JSON response."""
    if not ADK_AVAILABLE:
        return text

    return Event(
        author=author,
        invocation_id=getattr(parent_context, "invocation_id", ""),
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        actions=EventActions(),
        timestamp=time.time(),
    )


class PostCallWorkflowAgent(BaseAgent):
    """ADK orchestrator that runs the post-call collections workflow."""

    async def _run_async_impl(self, parent_context: Any):
        """Run the post-call workflow and yield a JSON-only ADK event."""
        borrower_profile, conversation_history, latest_message, final_outcome = (
            _context_payload(parent_context)
        )
        result = run_post_call_workflow_json(
            borrower_profile=borrower_profile,
            conversation_history=conversation_history,
            latest_borrower_message=latest_message,
            final_outcome=final_outcome,
        )
        yield _adk_event(parent_context, self.name, result)

    async def _run_live_impl(self, parent_context: Any):
        """Handle live invocations through the same workflow path."""
        async for event in self._run_async_impl(parent_context):
            yield event


root_agent = PostCallWorkflowAgent(
    name="post_call_workflow",
    description=(
        "Runs borrower understanding, supervisor routing, specialist reasoning, "
        "compliance checking, and final CRM tools for a post-call workflow."
    ),
)


__all__ = [
    "PostCallWorkflowAgent",
    "run_post_call_workflow",
    "run_post_call_workflow_json",
    "root_agent",
]

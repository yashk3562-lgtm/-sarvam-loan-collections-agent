from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from copy import deepcopy
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip() or st.secrets.get("SARVAM_API_KEY", "").strip()
BASE_URL = "https://api.sarvam.ai"

CHAT_URL = f"{BASE_URL}/v1/chat/completions"
STT_URL = f"{BASE_URL}/speech-to-text"
TTS_URL = f"{BASE_URL}/text-to-speech"

CHAT_MODEL = "sarvam-30b"
STT_MODEL = "saaras:v3"
TTS_MODEL = "bulbul:v3"
APP_VERSION = "presentation-poc-v6-fast-model-led"

LANGUAGE_CODES = {
    "Hindi/Hinglish": "hi-IN",
    "English": "en-IN",
    "Tamil": "ta-IN",
}

SPEAKERS = {
    "Hindi/Hinglish": "shubh",
    "English": "anand",
    "Tamil": "shreya",
}

BORROWERS: dict[str, dict[str, Any]] = {
    "BRW001": {
        "name": "Ramesh Kumar",
        "city": "Lucknow",
        "product": "Two-wheeler loan",
        "emi_amount": 4850,
        "overdue_days": 7,
        "risk": "Medium",
        "previous_attempts": 2,
    },
    "BRW002": {
        "name": "Priya Sharma",
        "city": "Bengaluru",
        "product": "Personal loan",
        "emi_amount": 9200,
        "overdue_days": 14,
        "risk": "High",
        "previous_attempts": 4,
    },
    "BRW003": {
        "name": "Suresh Iyer",
        "city": "Chennai",
        "product": "Consumer durable loan",
        "emi_amount": 3150,
        "overdue_days": 5,
        "risk": "Low",
        "previous_attempts": 1,
    },
}

DEFAULT_STATE: dict[str, Any] = {
    "messages": [],
    "crm_events": [],
    "reminders": [],
    "escalations": [],
    "analytics": {
        "Calls Started": 0,
        "Promise-to-Pay": 0,
        "Escalations": 0,
        "Restructure Requests": 0,
    },
    "call_started": False,
    "call_ended": False,
    "last_transcript": "",
    "last_audio_hash": "",
    "pending_confirmation": {},
    "last_agent_audio_b64": "",
    "last_audio_message_index": -1,
    "audio_rendered_for_index": -1,
    "pending_audio_autoplay": False,
    "no_commitment_count": 0,
    "hardship_refusal_count": 0,
    "last_agent_turn": {},
    "last_agent_parse_error": "",
}

st.set_page_config(
    page_title="Sarvam Collections Voice Agent",
    page_icon="🏦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }
    .hero-card {
        background: linear-gradient(135deg,#101827 0%,#312e81 55%,#4f46e5 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 24px;
        margin-bottom: 20px;
        box-shadow: 0 12px 32px rgba(15,23,42,.22);
    }
    .hero-card h1 {
        margin:0;
        color:white;
        font-size:36px;
        font-weight:900;
        letter-spacing:-.03em;
    }
    .hero-card p {
        color:#e5e7eb;
        margin-top:8px;
        margin-bottom:0;
        font-size:15px;
    }
    .section-card {
        border:1px solid rgba(148,163,184,.22);
        border-radius:18px;
        padding:18px;
        background:rgba(15,23,42,.35);
        box-shadow:0 4px 18px rgba(15,23,42,.08);
        margin-bottom:16px;
    }
    [data-testid="stMetric"] {
        background:rgba(15,23,42,.45);
        border:1px solid rgba(148,163,184,.25);
        border-radius:16px;
        padding:14px;
        box-shadow:0 4px 16px rgba(15,23,42,.12);
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] div {
        color:#f8fafc !important;
    }
    .stButton button {
        border-radius:12px !important;
        font-weight:700 !important;
    }
    div[data-testid="stAlert"] {
        border-radius:12px;
    }
    .call-guide {
        border: 1px solid rgba(148,163,184,.25);
        background: rgba(15,23,42,.45);
        padding: 14px 16px;
        border-radius: 14px;
        margin-bottom: 16px;
        color: #e5e7eb;
    }
    .call-guide strong { color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = deepcopy(value)


def reset_call() -> None:
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = deepcopy(value)


init_state()


def require_api_key() -> None:
    if not SARVAM_API_KEY:
        st.error("Missing SARVAM_API_KEY. Add it to voice_agent/.env and restart Streamlit.")
        st.stop()


def sarvam_headers(json_mode: bool = True) -> dict[str, str]:
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Authorization": f"Bearer {SARVAM_API_KEY}",
    }
    if json_mode:
        headers["Content-Type"] = "application/json"
    return headers


def parse_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        return payload.get("message") or payload.get("error") or response.text
    except Exception:
        return response.text


def extract_json_object(text: str) -> dict[str, Any]:
    if not text or not isinstance(text, str):
        raise ValueError("Empty JSON response")

    cleaned = text.strip()
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {text[:300]}")

    candidate = cleaned[start:end + 1]
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as first_error:
        repaired = candidate
        repaired = (
            repaired.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
        )
        repaired = re.sub(r"(?<=[{,])\s*'([^'{}:\[\],]+)'\s*:", r'"\1":', repaired)
        repaired = re.sub(r":\s*'([^'\\]*(?:\\.[^'\\]*)*)'", lambda match: ': ' + json.dumps(match.group(1)), repaired)
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)
        repaired = re.sub(r"\bNone\b", "null", repaired)
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError as second_error:
            raise ValueError(f"Could not parse JSON response: {second_error}") from first_error


def repair_json_with_sarvam(raw_text: str, user_text: str) -> dict[str, Any]:
    require_api_key()

    repair_messages = [
        {
            "role": "system",
            "content": (
                "Convert raw model output into valid JSON only. No markdown, no explanation. "
                "Use exactly these keys: reply, outcome, risk_score, next_action, promise_date, "
                "needs_confirmation, call_ended."
            ),
        },
        {
            "role": "user",
            "content": (
                "Raw model output:\n"
                f"{raw_text}\n\n"
                "Latest borrower message for context:\n"
                f"{user_text}\n\n"
                "Return valid JSON only with keys: reply, outcome, risk_score, next_action, "
                "promise_date, needs_confirmation, call_ended."
            ),
        },
    ]

    repair_payload = {
        "model": CHAT_MODEL,
        "messages": repair_messages,
        "temperature": 0.05,
        "max_tokens": 180,
        "reasoning_effort": "low",
    }

    repair_response = requests.post(
        CHAT_URL,
        headers=sarvam_headers(True),
        json=repair_payload,
        timeout=45,
    )

    if repair_response.status_code >= 400:
        raise RuntimeError(f"Sarvam JSON repair failed: {repair_response.status_code} - {parse_error(repair_response)}")

    repair_data = repair_response.json()
    repair_choices = repair_data.get("choices") or []
    if not repair_choices:
        raise RuntimeError(f"Unexpected JSON repair response: {repair_data}")

    repair_message = repair_choices[0].get("message") or {}
    repair_content = repair_message.get("content") or ""
    return extract_json_object(str(repair_content))


def sarvam_stt(audio_file: Any) -> str:
    require_api_key()

    files = {
        "file": ("borrower_response.wav", audio_file.getvalue(), "audio/wav"),
    }
    data = {
        "model": STT_MODEL,
        "mode": "codemix",
    }

    response = requests.post(
        STT_URL,
        headers=sarvam_headers(False),
        files=files,
        data=data,
        timeout=90,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"Sarvam STT failed: {response.status_code} - {parse_error(response)}")

    payload = response.json()
    return (
        payload.get("transcript")
        or payload.get("text")
        or payload.get("transcription")
        or ""
    ).strip()


#
# Helper function to detect meta or instruction replies
def is_meta_or_instruction_reply(reply: str) -> bool:
    lower = (reply or "").lower().strip()
    blocked = [
        "reason about",
        "reason about the borrower",
        "borrower's situation",
        "borrower situation",
        "conversation history",
        "agent sentence only",
        "no json",
        "no labels",
        "no reasoning",
        "return only",
        "valid json",
        "schema",
        "business objective",
        "conversation behavior",
        "next best response",
        "use the conversation",
        "do not use",
        "reply in the borrower's language",
        "keep the reply",
        "the agent should say",
        "exact sentence",
        "my task",
        "task:",
        "goal:",
        "rule:",
        "rules:",
        "instruction",
        "classification",
        "scratchpad",
        "analysis",
    ]
    return any(term in lower for term in blocked)

# New agent turn normalization and generation logic
def normalize_agent_turn(payload: dict[str, Any]) -> dict[str, Any]:
    reply = str(payload.get("reply", "")).strip()
    outcome = str(payload.get("outcome", "Continue")).strip()
    promise_date = str(payload.get("promise_date", "")).strip()
    next_action = str(payload.get("next_action", "Continue conversation")).strip()

    needs_confirmation = bool(payload.get("needs_confirmation", False))
    call_ended = bool(payload.get("call_ended", False))

    try:
        risk_score = int(payload.get("risk_score", 60))
    except Exception:
        risk_score = 60

    if not reply:
        reply = "Samajh gaya Ramesh ji. Aap payment ke liye realistic date bata sakte hain?"

    banned = [
        "analysis", "reasoning", "scratchpad", "output", "rule", "task",
        "borrower-facing", "message.content", "policy", "instruction", "classification",
        "schema", "valid json", "return only", "reason about", "borrower's situation",
        "conversation behavior", "business objective", "agent sentence only", "no reasoning"
    ]
    lower = reply.lower()
    if any(term in lower for term in banned) or is_meta_or_instruction_reply(reply):
        reply = "Samajh gaya Ramesh ji. Aapki baat samajh raha hoon. Aap next practical step kya suggest karenge?"

    reply = re.sub(r"<final>|</final>", "", reply, flags=re.IGNORECASE).strip()
    reply = re.sub(r"^(agent|assistant|reply|response)\s*[:\-]\s*", "", reply, flags=re.IGNORECASE).strip()
    risk_score = max(0, min(100, risk_score))

    valid_outcomes = {"Continue", "Promise Date Captured", "Promise-to-Pay", "Restructure Requested", "Escalation"}
    if outcome not in valid_outcomes:
        outcome = "Continue"

    return {
        "reply": reply,
        "outcome": outcome,
        "risk_score": risk_score,
        "next_action": next_action,
        "promise_date": promise_date,
        "needs_confirmation": needs_confirmation,
        "call_ended": call_ended,
    }



def technical_fallback_turn(account: dict[str, Any] | None = None) -> dict[str, Any]:
    first_name = (account or {}).get("name", "customer").split()[0]
    return normalize_agent_turn({
        "reply": f"Samajh gaya {first_name} ji. Aaj nahi ho paayega, toh kya koi chhota partial payment possible hai?",
        "outcome": "Continue",
        "risk_score": 65,
        "next_action": "Continue conversation after technical fallback",
        "promise_date": "",
        "needs_confirmation": False,
        "call_ended": False,
    })


def fallback_agent_turn(user_text: str, account: dict[str, Any] | None = None) -> dict[str, Any]:
    return technical_fallback_turn(account)


def apply_business_guardrails(turn: dict[str, Any], user_text: str, account: dict[str, Any]) -> dict[str, Any]:
    guarded = dict(turn)
    text = (user_text or "").lower()
    first_name = account["name"].split()[0]

    explicit_escalation_terms = [
        "human", "manager", "supervisor", "fraud", "wrong loan", "not my loan",
        "complaint", "harassment", "legal", "court", "police",
    ]
    cannot_pay_terms = [
        "cannot pay", "can't pay", "nahi kar sakta", "nahi kar paunga",
        "payment nahi", "aaj nahi", "paise nahi", "funds available nahi", "funds available नहीं",
    ]
    job_loss_terms = [
        "job chali", "job nahi", "job नहीं", "naukri nahi", "नौकरी नहीं", "नौकरी चली",
        "no job", "lost job", "job gone", "unemployed", "salary nahi", "salary नहीं",
    ]
    no_partial_terms = [
        "partial payment possible nahi", "partial payment possible नहीं", "partial payment nahi",
        "partial payment नहीं", "chhota payment nahi", "छोटा payment नहीं", "कुछ नहीं दे सकता",
        "kuch nahi de sakta", "bilkul bhi nahi", "बिल्कुल भी नहीं", "not possible", "possible nahi", "possible नहीं",
    ]
    no_date_terms = [
        "date nahi", "date नहीं", "kab nahi pata", "कब नहीं पता", "pata nahi", "पता नहीं",
        "no idea", "don't know", "dont know", "cannot commit", "commit nahi", "commit नहीं",
        "unable to give date", "date of payment nahi", "payment date nahi",
    ]
    date_like_terms = [
        "monday", "next week", "next month", "month end", "salary",
        "tomorrow", "kal", "parso", "tarikh", "तारीख",
    ]

    has_explicit_escalation = any(term in text for term in explicit_escalation_terms)
    has_job_loss = any(term in text for term in job_loss_terms)
    has_cannot_pay = any(term in text for term in cannot_pay_terms) or has_job_loss
    has_date_like = any(term in text for term in date_like_terms)
    has_no_partial = any(term in text for term in no_partial_terms)
    has_no_date = any(term in text for term in no_date_terms)

    if guarded.get("outcome") == "Escalation" and not has_explicit_escalation:
        guarded["outcome"] = "Continue"
        guarded["call_ended"] = False
        guarded["needs_confirmation"] = False
        guarded["next_action"] = "Continue conversation"

    if has_no_partial or has_no_date:
        st.session_state.hardship_refusal_count = st.session_state.get("hardship_refusal_count", 0) + 1
    else:
        st.session_state.hardship_refusal_count = 0

    if has_no_partial and (has_no_date or st.session_state.get("hardship_refusal_count", 0) >= 2):
        guarded.update({
            "reply": f"Samajh gaya {first_name} ji. Main human team se callback arrange kar deta hoon, woh hardship options discuss karenge.",
            "outcome": "Escalation",
            "risk_score": max(85, int(guarded.get("risk_score", 85) or 85)),
            "next_action": "Human callback for no partial payment and no payment date",
            "promise_date": "",
            "needs_confirmation": False,
            "call_ended": True,
        })
        return normalize_agent_turn(guarded)

    if has_job_loss:
        guarded.update({
            "reply": f"Samajh gaya {first_name} ji. Job issue tough hai. Kya family support, alternate income, ya small partial payment possible hai?",
            "outcome": "Continue",
            "risk_score": max(70, int(guarded.get("risk_score", 70) or 70)),
            "next_action": "Explore hardship support, alternate income, or partial payment",
            "promise_date": "",
            "needs_confirmation": False,
            "call_ended": False,
        })
        return normalize_agent_turn(guarded)

    if has_cannot_pay:
        guarded.update({
            "reply": f"Samajh gaya {first_name} ji. Aaj nahi ho paayega, toh kya koi chhota partial payment possible hai?",
            "outcome": "Continue",
            "risk_score": max(65, int(guarded.get("risk_score", 65) or 65)),
            "next_action": "Ask partial payment option",
            "promise_date": "",
            "needs_confirmation": False,
            "call_ended": False,
        })

    if has_date_like and not has_job_loss:
        promise_date = extract_date_hint(user_text)
        guarded.update({
            "reply": f"Theek hai. Kya main {promise_date} ke liye payment reminder confirm kar doon?",
            "outcome": "Promise Date Captured",
            "next_action": "Confirm payment reminder",
            "promise_date": promise_date,
            "needs_confirmation": True,
            "call_ended": False,
        })

    if guarded.get("outcome") not in ["Promise-to-Pay", "Escalation"]:
        guarded["call_ended"] = False

    return normalize_agent_turn(guarded)


def build_agent_system_prompt(account: dict[str, Any], language: str, pending: dict[str, Any]) -> str:
    return f"""
You are the conversation brain for a compliant Indian loan collections voice agent.

Borrower:
Name: {account['name']}
City: {account['city']}
Loan: {account['product']}
Overdue EMI: INR {account['emi_amount']}
Days overdue: {account['overdue_days']}
Risk: {account['risk']}
Language: {language}
Pending confirmation: {pending or 'None'}

Business objective:
Recover the overdue EMI, get a realistic promise-to-pay, offer partial payment if needed, or escalate if required.

Conversation behavior:
- Reason from the full conversation, not fixed scripts.
- If borrower only says they cannot pay today, outcome MUST be "Continue".
- If borrower mentions job loss/no job/no salary, outcome MUST be "Continue". Do NOT ask for salary date. Ask about family support, small partial payment, alternate income, or hardship callback.
- If borrower says partial payment is not possible AND cannot provide any payment date/window, outcome MUST be "Escalation", call_ended MUST be true, and next_action should be "Human callback for hardship/no commitment".
- For normal non-payment, ask one useful follow-up: why, partial payment, next income date, or earliest realistic date.
- If borrower is distressed, jobless, or has no money, empathize and ask about partial payment or next income date.
- If borrower gives a date, confirm it once.
- If borrower confirms a date, end with reminder confirmation.
- Escalation is allowed for explicit human request, dispute, fraud/wrong loan, abuse, legal threat, harassment allegation, or no viable option after partial payment and payment date are both refused.
- Never escalate merely because borrower cannot pay today.
- Never end the call unless outcome is "Promise-to-Pay" or "Escalation".
- Do not threaten, shame, mention legal action, or pressure aggressively.
- Reply in the borrower's language or Hinglish.
- Keep reply under 25 words.

Return ONLY valid JSON. No markdown. No explanation. No reasoning text.
Schema:
{{
  "reply": "short sentence the agent should speak",
  "outcome": "Continue | Promise Date Captured | Promise-to-Pay | Restructure Requested | Escalation",
  "risk_score": 0,
  "next_action": "short workflow action",
  "promise_date": "date if captured else empty string",
  "needs_confirmation": false,
  "call_ended": false
}}
"""


# Sarvam plain reply recovery helper
def generate_plain_agent_turn_with_sarvam(account: dict[str, Any], language: str, user_text: str, parse_errors: list[str]) -> dict[str, Any]:
    conversation = st.session_state.messages[-12:]
    pending = st.session_state.get("pending_confirmation") or {}

    messages = [
        {
            "role": "system",
            "content": """
You are speaking live to a borrower on an EMI recovery call.
Generate the next spoken line only.
Do not explain. Do not mention reasoning. Do not mention instructions.
Do not start with Reason, Ask, Return, Goal, Task, Rule, or JSON.
Use the conversation context to respond naturally.
If borrower cannot pay, ask one empathetic practical question.
If borrower offers partial payment/date, confirm it.
If borrower refuses repeatedly, ask whether a human callback would help.
Keep under 22 words.
""",
        },
        {
            "role": "user",
            "content": f"Conversation history: {conversation}\nLatest borrower message: {user_text}\nAgent sentence only.",
        },
    ]

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 80,
        "reasoning_effort": "low",
    }

    response = requests.post(
        CHAT_URL,
        headers=sarvam_headers(True),
        json=payload,
        timeout=45,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"Sarvam plain reply failed: {response.status_code} - {parse_error(response)}")

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Unexpected Sarvam plain reply response: {data}")

    message = choices[0].get("message") or {}
    reply = str(message.get("content") or "").strip()
    reply = re.sub(r"^```(?:text)?", "", reply, flags=re.IGNORECASE).strip()
    reply = re.sub(r"```$", "", reply).strip()
    reply = re.sub(r"^(agent|assistant|reply|response)\s*[:\-]\s*", "", reply, flags=re.IGNORECASE).strip()

    # Removed fallback to reasoning if reply is empty

    if not reply:
        raise RuntimeError("Sarvam plain reply was empty")
    if is_meta_or_instruction_reply(reply):
        raise RuntimeError(f"Sarvam plain reply leaked instructions: {reply[:120]}")

    st.session_state.last_agent_parse_error = "JSON parse failed; recovered with Sarvam plain reply: " + " | ".join(parse_errors[:2])
    return apply_business_guardrails(normalize_agent_turn({
        "reply": reply,
        "outcome": "Continue",
        "risk_score": 65,
        "next_action": "Continue model-led conversation",
        "promise_date": "",
        "needs_confirmation": False,
        "call_ended": False,
    }), user_text, account)


def generate_agent_turn(account: dict[str, Any], language: str, user_text: str) -> dict[str, Any]:
    require_api_key()

    conversation = st.session_state.messages[-10:]
    pending = st.session_state.get("pending_confirmation") or {}

    text = user_text.lower()
    confirmation_terms = [
        "haan", "yes", "ok", "okay", "confirm", "confirmed",
        "theek", "thik", "kar sakta", "kar paunga", "message",
        "call", "reminder", "haan ji", "हाँ", "ठीक", "कर सकता"
    ]

    if pending and any(term in text for term in confirmation_terms):
        return normalize_agent_turn({
            "reply": f"Theek hai {account['name'].split()[0]} ji. Main {pending.get('promise_date', 'is date')} ke liye reminder set kar deta hoon.",
            "outcome": "Promise-to-Pay",
            "risk_score": 35,
            "next_action": "Reminder queued",
            "promise_date": pending.get("promise_date", "Customer committed date"),
            "needs_confirmation": False,
            "call_ended": True,
        })

    system = build_agent_system_prompt(account, language, pending)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Conversation so far: {conversation}\nLatest borrower message: {user_text}\nReturn JSON only."},
    ]

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.15,
        "max_tokens": 240,
        "reasoning_effort": "low",
    }

    response = requests.post(
        CHAT_URL,
        headers=sarvam_headers(True),
        json=payload,
        timeout=75,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"Sarvam chat failed: {response.status_code} - {parse_error(response)}")

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Unexpected chat response: {data}")

    message = choices[0].get("message") or {}
    content = message.get("content") or ""

    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if isinstance(reasoning, dict):
        reasoning = reasoning.get("content") or reasoning.get("text") or ""

    raw_candidates = [str(raw) for raw in [content, reasoning] if str(raw or "").strip()]
    parse_errors = []

    for index, raw in enumerate(raw_candidates, start=1):
        try:
            turn = normalize_agent_turn(extract_json_object(raw))
            if is_meta_or_instruction_reply(turn["reply"]):
                raise ValueError(f"meta reply rejected: {turn['reply'][:120]}")
            return apply_business_guardrails(turn, user_text, account)
        except Exception as exc:
            parse_errors.append(f"candidate {index} parse failed: {exc}")

    for index, raw in enumerate(raw_candidates, start=1):
        try:
            turn = normalize_agent_turn(repair_json_with_sarvam(raw, user_text))
            if is_meta_or_instruction_reply(turn["reply"]):
                raise ValueError(f"meta repair reply rejected: {turn['reply'][:120]}")
            return apply_business_guardrails(turn, user_text, account)
        except Exception as exc:
            parse_errors.append(f"candidate {index} repair failed: {exc}")

    try:
        return generate_plain_agent_turn_with_sarvam(account, language, user_text, parse_errors)
    except Exception as exc:
        parse_errors.append(f"plain Sarvam recovery failed: {exc}")

    st.session_state.last_agent_parse_error = " | ".join(parse_errors)
    return apply_business_guardrails(fallback_agent_turn(user_text, account), user_text, account)


def sarvam_tts(text: str, language: str) -> bytes | None:
    require_api_key()

    payload = {
        "text": text[:1000],
        "target_language_code": LANGUAGE_CODES.get(language, "hi-IN"),
        "speaker": SPEAKERS.get(language, "shubh"),
        "model": TTS_MODEL,
        "pace": 1.08,
        "speech_sample_rate": 24000,
        "output_audio_codec": "wav",
        "temperature": 0.3,
    }

    response = requests.post(
        TTS_URL,
        headers=sarvam_headers(True),
        json=payload,
        timeout=90,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"Sarvam TTS failed: {response.status_code} - {parse_error(response)}")

    payload = response.json()
    audios = payload.get("audios") or []

    if not audios:
        return None

    return base64.b64decode(audios[0])


def opening_line(account: dict[str, Any], language: str) -> str:
    first_name = account["name"].split()[0]
    amount = account["emi_amount"]

    if language == "English":
        return f"Hello {first_name}. Your EMI of rupees {amount} is overdue. Can you pay today?"

    if language == "Tamil":
        return f"Vanakkam {first_name}. Ungal {amount} rupai EMI overdue. Indru payment panna mudiyuma?"

    return f"Namaste {first_name} ji. Aapka {amount} rupaye EMI overdue hai. Kya aaj payment kar sakte hain?"


def extract_date_hint(text: str) -> str:
    normalized = text.lower()

    if "8th" in normalized or "8 july" in normalized or "8th of july" in normalized:
        return "8th July"
    if "next month" in normalized and ("15" in normalized or "15th" in normalized or "15 तारीख" in normalized or "15 tarikh" in normalized):
        return "15th next month"
    if "15 तारीख" in normalized or "15 tarikh" in normalized:
        return "15th"

    if "next month end" in normalized or "month end" in normalized:
        return "next month end"
    if "next month" in normalized:
        return "next month"

    if "monday" in normalized or "सोमवार" in normalized:
        return "Monday"

    if "today" in normalized:
        return "today"

    if "tomorrow" in normalized or "kal" in normalized or "कल" in normalized:
        return "tomorrow"

    if "parso" in normalized or "परसों" in normalized:
        return "day after tomorrow"

    if "salary" in normalized:
        return "salary day"

    for hint in ["friday", "tuesday", "wednesday", "thursday", "saturday", "sunday", "10th", "15th"]:
        if hint in normalized:
            return hint

    return "Customer committed date"


def classify_outcome(user_text: str, agent_reply: str) -> dict[str, Any]:
    return st.session_state.get("last_agent_turn", {
        "outcome": "Continue",
        "risk_score": 60,
        "next_action": "Continue conversation",
        "promise_date": "",
        "call_ended": False,
        "needs_confirmation": False,
    })


def apply_workflow(account_id: str, account: dict[str, Any], user_text: str, agent_reply: str) -> None:
    outcome = classify_outcome(user_text, agent_reply)

    if outcome.get("needs_confirmation"):
        st.session_state.pending_confirmation = {"promise_date": outcome.get("promise_date", "captured date")}
        return

    if outcome["outcome"] not in ["Promise-to-Pay", "Escalation"]:
        return

    if st.session_state.crm_events:
        return

    if outcome["outcome"] == "Promise-to-Pay":
        st.session_state.pending_confirmation = {}
        st.session_state.reminders.append(
            {
                "borrower": account_id,
                "promise_date": outcome["promise_date"],
                "channel": "WhatsApp/SMS",
                "status": "Queued",
            }
        )
        st.session_state.analytics["Promise-to-Pay"] += 1

    if outcome["outcome"] == "Escalation":
        st.session_state.escalations.append(
            {
                "borrower": account_id,
                "reason": outcome.get("next_action", "Human callback required"),
                "owner": "Human collections manager",
                "status": "Open",
            }
        )
        st.session_state.analytics["Escalations"] += 1

    st.session_state.crm_events.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "borrower": account_id,
            "customer": account["name"],
            "outcome": outcome["outcome"],
            "risk_score": outcome["risk_score"],
            "next_action": outcome["next_action"],
        }
    )

    st.session_state.call_ended = True


def generate_reply(account: dict[str, Any], language: str, user_text: str) -> str:
    turn = generate_agent_turn(account, language, user_text)
    st.session_state.last_agent_turn = turn
    return turn["reply"]


def play_agent_audio(text: str, language: str, message_index: int | None = None) -> None:
    audio_bytes = sarvam_tts(text, language)

    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        st.session_state.last_agent_audio_b64 = b64
        st.session_state.last_audio_message_index = message_index if message_index is not None else len(st.session_state.messages) - 1
        st.session_state.pending_audio_autoplay = True


def render_latest_agent_audio() -> None:
    idx = st.session_state.get("last_audio_message_index", -1)
    audio_b64 = st.session_state.get("last_agent_audio_b64", "")

    if not audio_b64 or idx < 0:
        return

    if not st.session_state.get("pending_audio_autoplay", False):
        return

    st.session_state.pending_audio_autoplay = False
    st.session_state.audio_rendered_for_index = idx
    st.markdown(
        f"""
        <audio id="latest-agent-audio" autoplay controls>
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        </audio>
        <script>
            const audio = document.getElementById("latest-agent-audio");
            if (audio) {{ audio.play().catch(() => {{}}); }}
        </script>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero-card">
        <h1>Sarvam AI Collections & Recovery Agent</h1>
        <p>Multilingual EMI recovery PoC using Sarvam Speech-to-Text, Sarvam 30B, and Bulbul Text-to-Speech. Build: presentation-poc-v6-fast-model-led.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Demo Controls")

    selected_borrower = st.selectbox("Borrower", list(BORROWERS.keys()))
    language = st.selectbox("Conversation Language", ["Hindi/Hinglish", "English", "Tamil"])
    voice_enabled = st.toggle("Play agent voice using Sarvam TTS", value=True)

    if SARVAM_API_KEY:
        st.success("Sarvam API key loaded")
    else:
        st.error("Missing SARVAM_API_KEY")

with st.expander("API diagnostics", expanded=False):
    st.caption(f"Build: {APP_VERSION}")
    st.caption(f"Chat: {CHAT_URL}")
    st.caption(f"STT: {STT_URL}")
    st.caption(f"TTS: {TTS_URL}")
    st.caption(f"Model: {CHAT_MODEL}")
    if st.session_state.get("last_agent_parse_error"):
        st.caption(f"Last agent parse recovery: {st.session_state.last_agent_parse_error}")

    if st.button("Reset Demo", use_container_width=True):
        reset_call()
        st.rerun()

account = BORROWERS[selected_borrower]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Borrower", account["name"])
m2.metric("Overdue EMI", f"₹{account['emi_amount']:,}")
m3.metric("Overdue Days", account["overdue_days"])
m4.metric("Risk", account["risk"])

left, right = st.columns([1.35, 1])

with left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Live Voice Conversation")
    st.markdown(
        """
        <div class="call-guide">
            <strong>Demo flow:</strong> Start Call → listen to agent → record borrower → stop recording → agent replies with voice → repeat.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.call_started:
        if st.button("Start Call", type="primary", use_container_width=True):
            st.session_state.call_started = True
            st.session_state.analytics["Calls Started"] += 1

            first_reply = opening_line(account, language)
            st.session_state.messages.append({"role": "assistant", "content": first_reply})

            if voice_enabled:
                with st.spinner("Generating Sarvam opening voice..."):
                    play_agent_audio(first_reply, language, len(st.session_state.messages) - 1)
            st.rerun()
    else:
        st.caption(
            "Call is active. Record the borrower response; processing starts automatically after recording stops."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            label = "Agent" if msg["role"] == "assistant" else "Borrower"
            st.write(f"**{label}:** {msg['content']}")

    render_latest_agent_audio()

    if st.session_state.call_ended:
        st.success("Call outcome captured. Conversation ended.")

    elif st.session_state.call_started:
        st.markdown("#### Your turn: record borrower response")
        st.caption(
            "Record, stop, and the app will auto-run: Sarvam STT → Sarvam 30B → Sarvam TTS → CRM update."
        )

        audio_input = st.audio_input("Record borrower response, then stop")

        if audio_input is not None:
            audio_bytes = audio_input.getvalue()
            audio_hash = hashlib.md5(audio_bytes).hexdigest()

            if audio_hash != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = audio_hash

                try:
                    with st.spinner("Sarvam Saaras is transcribing borrower audio..."):
                        transcript = sarvam_stt(audio_input)

                    if not transcript:
                        st.warning("No transcript detected. Record again with clearer audio.")
                    else:
                        st.session_state.last_transcript = transcript
                        st.session_state.messages.append({"role": "user", "content": transcript})

                        with st.spinner("Sarvam 30B is generating the agent response..."):
                            reply = generate_reply(account, language, transcript)

                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        apply_workflow(selected_borrower, account, transcript, reply)

                        if voice_enabled:
                            with st.spinner("Generating Sarvam agent voice..."):
                                play_agent_audio(reply, language, len(st.session_state.messages) - 1)

                        st.rerun()

                except Exception as exc:
                    st.error(f"Voice response failed: {exc}")

        if st.session_state.last_transcript:
            st.caption(f"Last borrower transcript: {st.session_state.last_transcript}")

        typed_response = st.chat_input("Backup: type borrower response")

        if typed_response:
            try:
                st.session_state.messages.append({"role": "user", "content": typed_response})

                with st.spinner("Sarvam 30B is generating response..."):
                    reply = generate_reply(account, language, typed_response)

                st.session_state.messages.append({"role": "assistant", "content": reply})
                apply_workflow(selected_borrower, account, typed_response, reply)

                if voice_enabled:
                    with st.spinner("Generating Sarvam agent voice..."):
                        play_agent_audio(reply, language, len(st.session_state.messages) - 1)

                st.rerun()

            except Exception as exc:
                st.error(f"Agent response failed: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Borrower Context")
    st.write(f"**Name:** {account['name']}")
    st.write(f"**City:** {account['city']}")
    st.write(f"**Loan:** {account['product']}")
    st.write(f"**Overdue EMI:** ₹{account['emi_amount']:,}")
    st.write(f"**Days overdue:** {account['overdue_days']}")
    st.write(f"**Previous attempts:** {account['previous_attempts']}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Live Analytics")
    st.dataframe(pd.DataFrame([st.session_state.analytics]), use_container_width=True, hide_index=True)

    risk_score = st.session_state.crm_events[-1]["risk_score"] if st.session_state.crm_events else 50
    st.progress(risk_score / 100, text=f"Current risk score: {risk_score}/100")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("CRM Events")
    if st.session_state.crm_events:
        st.dataframe(pd.DataFrame(st.session_state.crm_events), use_container_width=True, hide_index=True)
    else:
        st.info("No CRM events yet.")

with c2:
    st.subheader("Reminder Queue")
    if st.session_state.reminders:
        st.dataframe(pd.DataFrame(st.session_state.reminders), use_container_width=True, hide_index=True)
    else:
        st.info("No reminders yet.")

with c3:
    st.subheader("Escalation Queue")
    if st.session_state.escalations:
        st.dataframe(pd.DataFrame(st.session_state.escalations), use_container_width=True, hide_index=True)
    else:
        st.info("No escalations yet.")

st.divider()

with st.expander("Architecture and production path"):
    st.markdown(
        """
        **PoC flow:** Browser microphone -> Sarvam `/speech-to-text` -> Sarvam `/v1/chat/completions` JSON agent turn -> Sarvam `/text-to-speech` -> CRM workflow.

        **Production flow:** Exotel/Twilio/LiveKit telephony -> streaming STT -> Sarvam 30B/105B -> workflow engine -> streaming TTS -> customer.
        """
    )

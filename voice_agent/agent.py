from __future__ import annotations

import base64
import hashlib
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

CHAT_MODEL = "sarvam-105b"
STT_MODEL = "saaras:v3"
TTS_MODEL = "bulbul:v3"
APP_VERSION = "presentation-poc-v5-105b-model-led"

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


def extract_spoken_reply_from_reasoning(reasoning_text: str) -> str:
    """Extract only the final spoken response if Sarvam returns it inside reasoning_content."""
    if not reasoning_text or not isinstance(reasoning_text, str):
        return ""

    final_match = re.search(r"<final>(.*?)</final>", reasoning_text, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        candidate = re.sub(r"\s+", " ", final_match.group(1)).strip()
        if candidate:
            return candidate

    patterns = [
        r"(?:final spoken reply|final answer|final response|assistant reply|agent reply|reply to borrower)[:\s\*]*[\"']([^\"']{3,220})[\"']",
        r"[\"']([^\"']{3,220})[\"']\s*(?:$|\n)",
    ]

    blocked_terms = [
        "analysis", "reasoning", "scratchpad", "rule", "policy", "simplest", "follows the rules",
        "my response", "should be", "let's", "lets", "user has", "borrower has", "if confirmed",
        "conversation goal", "output rule", "current state", "business goal"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, reasoning_text, flags=re.IGNORECASE | re.DOTALL)
        for candidate in reversed(matches):
            candidate = re.sub(r"\s+", " ", candidate).strip()
            candidate = candidate.strip(" -*\n\t\r\"")
            lower = candidate.lower()
            if not candidate:
                continue
            if any(blocked in lower for blocked in blocked_terms):
                continue
            if 3 <= len(candidate) <= 220:
                return candidate

    return ""


def sarvam_chat(messages: list[dict[str, str]]) -> str:
    require_api_key()

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.25,
        "max_tokens": 900,
        "reasoning_effort": "low",
    }

    response = requests.post(
        CHAT_URL,
        headers=sarvam_headers(True),
        json=payload,
        timeout=90,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Sarvam chat failed at {CHAT_URL}: {response.status_code} - {parse_error(response)}"
        )

    data = response.json()
    choices = data.get("choices") or []

    if not choices:
        raise RuntimeError(f"Unexpected chat response: {data}")

    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, str) and content.strip():
        cleaned_content = content.strip()
        final_match = re.search(r"<final>(.*?)</final>", cleaned_content, flags=re.IGNORECASE | re.DOTALL)
        if final_match:
            return re.sub(r"\s+", " ", final_match.group(1)).strip()
        return cleaned_content.replace("<final>", "").replace("</final>", "").strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return " ".join(parts).strip()

    reasoning_text = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning_text, dict):
        reasoning_text = reasoning_text.get("content") or reasoning_text.get("text")
    if isinstance(reasoning_text, str):
        spoken_reply = extract_spoken_reply_from_reasoning(reasoning_text)
        if spoken_reply:
            return spoken_reply

    raise RuntimeError("Sarvam returned reasoning without a clean final spoken reply. Try recording again or use the typed backup input.")


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


def system_prompt(account: dict[str, Any], language: str) -> str:
    pending = st.session_state.get("pending_confirmation") or {}
    pending_text = "No pending commitment."

    if pending:
        pending_text = (
            "The borrower has mentioned a possible promise-to-pay date: "
            f"{pending.get('promise_date', 'captured date')}. Your next goal is to confirm it naturally."
        )

    return f"""
You are a multilingual AI collections agent for an Indian NBFC. You are speaking to the borrower on a live recovery call.

Borrower profile:
- Name: {account['name']}
- City: {account['city']}
- Product: {account['product']}
- Overdue EMI: INR {account['emi_amount']}
- Days overdue: {account['overdue_days']}
- Risk: {account['risk']}
- Conversation language: {language}
- Current state: {pending_text}

Your business goal:
Recover the overdue EMI while keeping the borrower comfortable and compliant.

Your conversation goals, in priority order:
1. Ask for payment today.
2. If today is not possible, understand why.
3. Ask for the earliest realistic payment date.
4. If full payment is difficult, explore partial payment or split payment.
5. If a date or amount is mentioned, confirm it once.
6. If confirmed, close politely and mention that a reminder will be sent.
7. If the borrower disputes the loan, asks for a human, or sounds upset, escalate politely.

Reasoning guidance:
Use the full conversation context. Do not rely on fixed scripts. Adapt to whatever the borrower says.
Infer intent from messy Hinglish/Hindi/English. Handle vague answers, excuses, affordability issues, salary delays, travel, disputes, and callback requests naturally.
Do not end the call just because the borrower refuses once. Continue with one useful follow-up question unless the borrower confirms a plan or asks to stop.

Compliance rules:
Never threaten, shame, harass, mention legal action, or use aggressive language.
Be polite, concise, and human.
Ask only one question at a time.
Keep each reply under 30 words.
Reply in the borrower’s language or Hinglish if the borrower uses Hinglish.

    Output rule:
    Return only the final spoken agent reply wrapped in <final> and </final>.
    Example: <final>Samajh gaya Ramesh ji. Aap payment kab tak kar paayenge?</final>
    Do not include analysis, reasoning, options, labels, markdown, bullets, JSON, or quotes outside the final tags.
    Do not say what the best response should be. Say the response itself.
    Never output phrases like "stick to", "follows the rules", "my response", "response should be", or any instruction text.
"""


def extract_date_hint(text: str) -> str:
    normalized = text.lower()

    if "8th" in normalized or "8 july" in normalized or "8th of july" in normalized:
        return "8th July"
    if "next month" in normalized and ("15" in normalized or "15th" in normalized or "15 तारीख" in normalized or "15 tarikh" in normalized):
        return "15th next month"
    if "15 तारीख" in normalized or "15 tarikh" in normalized:
        return "15th"

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
    text = user_text.lower()

    confirmation_terms = [
        "haan", "yes", "confirm", "ok", "okay", "theek", "ठीक", "हाँ", "ha", "kar do", "done", "sahi hai"
    ]
    date_terms = [
        "today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "kal", "parso", "next week", "next month", "salary", "month end", "सोमवार", "कल", "परसों",
        "तारीख", "tarikh", "8th", "10th", "15th", "15 तारीख", "15 tarikh", "july"
    ]
    escalation_terms = [
        "fraud", "legal", "complaint", "harassment", "wrong loan", "dispute", "not my loan", "galat", "human", "manager", "representative"
    ]
    restructure_terms = [
        "split", "partial", "part payment", "installment", "extension", "cannot pay full", "full nahi", "time chahiye", "emi reduce"
    ]

    if any(term in text for term in escalation_terms):
        return {"outcome": "Escalation", "risk_score": 90, "next_action": "Human collections callback", "promise_date": "", "call_ended": True, "needs_confirmation": False}

    if any(term in text for term in restructure_terms):
        return {"outcome": "Restructure Requested", "risk_score": 72, "next_action": "Explore partial/split payment", "promise_date": "To be confirmed", "call_ended": False, "needs_confirmation": False}

    if st.session_state.pending_confirmation and any(term in text for term in confirmation_terms):
        return {"outcome": "Promise-to-Pay", "risk_score": 35, "next_action": "Reminder queued", "promise_date": st.session_state.pending_confirmation.get("promise_date", "Customer committed date"), "call_ended": True, "needs_confirmation": False}

    if any(term in text for term in date_terms):
        return {"outcome": "Promise Date Captured", "risk_score": 45, "next_action": "Confirm payment date", "promise_date": extract_date_hint(text), "call_ended": False, "needs_confirmation": True}

    return {"outcome": "Follow-up Needed", "risk_score": 60, "next_action": "Continue conversation", "promise_date": "", "call_ended": False, "needs_confirmation": False}


def apply_workflow(account_id: str, account: dict[str, Any], user_text: str, agent_reply: str) -> None:
    outcome = classify_outcome(user_text, agent_reply)

    if outcome.get("needs_confirmation"):
        st.session_state.pending_confirmation = {"promise_date": outcome["promise_date"]}
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
                "reason": "Dispute/refusal/callback requested",
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
    messages = [{"role": "system", "content": system_prompt(account, language)}]
    messages.extend(st.session_state.messages[-8:])
    messages.append({"role": "user", "content": user_text})
    return sarvam_chat(messages)


def play_agent_audio(text: str, language: str, message_index: int | None = None) -> None:
    audio_bytes = sarvam_tts(text, language)

    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode("utf-8")
        st.session_state.last_agent_audio_b64 = b64
        st.session_state.last_audio_message_index = message_index if message_index is not None else len(st.session_state.messages) - 1


def render_latest_agent_audio() -> None:
    idx = st.session_state.get("last_audio_message_index", -1)
    audio_b64 = st.session_state.get("last_agent_audio_b64", "")

    if not audio_b64 or idx < 0:
        return

    if st.session_state.get("audio_rendered_for_index", -1) == idx:
        return

    st.session_state.audio_rendered_for_index = idx
    st.markdown(
        f"""
        <audio autoplay controls>
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero-card">
        <h1>Sarvam AI Collections & Recovery Agent</h1>
        <p>Multilingual EMI recovery PoC using Sarvam Speech-to-Text, Sarvam 105B, and Bulbul Text-to-Speech. Build: presentation-poc-v5-105b-model-led.</p>
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
            "Record, stop, and the app will auto-run: Sarvam STT → Sarvam 105B → Sarvam TTS → CRM update."
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

                        with st.spinner("Sarvam 105B is generating the agent response..."):
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

                with st.spinner("Sarvam 105B is generating response..."):
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
        **PoC flow:** Browser microphone -> Sarvam `/speech-to-text` -> Sarvam `/v1/chat/completions` -> Sarvam `/text-to-speech` -> CRM workflow.

        **Production flow:** Exotel/Twilio/LiveKit telephony -> streaming STT -> Sarvam 30B/105B -> workflow engine -> streaming TTS -> customer.
        """
    )
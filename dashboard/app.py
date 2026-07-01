from __future__ import annotations

import base64
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Sarvam Collections Agent",
    layout="wide",
)

# =========================
# Sarvam Config
# =========================

SARVAM_API_KEY = st.secrets.get("SARVAM_API_KEY", "")

CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"
TTS_URL = "https://api.sarvam.ai/text-to-speech"
STT_URL = "https://api.sarvam.ai/speech-to-text"

LANG_CODES = {
    "Hindi": "hi-IN",
    "English": "en-IN",
    "Hinglish": "hi-IN",
    "Tamil": "ta-IN",
    "Kannada": "kn-IN",
    "Marathi": "mr-IN",
}

SPEAKERS = {
    "Hindi": "shubh",
    "English": "anand",
    "Hinglish": "shubh",
    "Tamil": "shreya",
    "Kannada": "anand",
    "Marathi": "shubh",
}


# =========================
# Mock Enterprise Data
# =========================

ACCOUNTS = {
    "BRW001": {
        "name": "Ramesh Kumar",
        "language": "Hindi",
        "city": "Lucknow",
        "product": "Two-wheeler loan",
        "emi_amount": 4850,
        "overdue_days": 7,
        "risk": "Medium",
        "attempts": 2,
    },
    "BRW002": {
        "name": "Priya Sharma",
        "language": "English",
        "city": "Bengaluru",
        "product": "Personal loan",
        "emi_amount": 9200,
        "overdue_days": 14,
        "risk": "High",
        "attempts": 4,
    },
    "BRW003": {
        "name": "Suresh Iyer",
        "language": "Tamil",
        "city": "Chennai",
        "product": "Consumer durable loan",
        "emi_amount": 3150,
        "overdue_days": 5,
        "risk": "Low",
        "attempts": 1,
    },
}


# =========================
# Session State
# =========================

def init_state() -> None:
    defaults = {
        "messages": [],
        "crm": [],
        "reminders": [],
        "escalations": [],
        "analytics": {
            "Calls": 0,
            "Promise-to-Pay": 0,
            "Escalations": 0,
            "Restructure Requests": 0,
        },
        "last_transcript": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_call() -> None:
    st.session_state.messages = []
    st.session_state.last_transcript = ""


init_state()


# =========================
# API Helpers
# =========================

def require_key() -> None:
    if not SARVAM_API_KEY:
        st.error("Missing SARVAM_API_KEY. Add it in Streamlit Secrets.")
        st.stop()


def sarvam_headers(json_mode: bool = True) -> dict[str, str]:
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Authorization": f"Bearer {SARVAM_API_KEY}",
    }
    if json_mode:
        headers["Content-Type"] = "application/json"
    return headers


def sarvam_chat(messages: list[dict[str, str]]) -> str:
    require_key()

    payload = {
        "model": "sarvam-105b",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 450,
        "reasoning_effort": "medium",
    }

    response = requests.post(
        CHAT_URL,
        headers=sarvam_headers(json_mode=True),
        json=payload,
        timeout=60,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"Chat API error {response.status_code}: {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]


def sarvam_tts(text: str, language: str) -> bytes | None:
    require_key()

    payload = {
        "text": text[:2400],
        "target_language_code": LANG_CODES.get(language, "hi-IN"),
        "speaker": SPEAKERS.get(language, "shubh"),
        "model": "bulbul:v3",
        "pace": 0.95,
        "speech_sample_rate": 24000,
        "output_audio_codec": "wav",
        "temperature": 0.4,
    }

    response = requests.post(
        TTS_URL,
        headers=sarvam_headers(json_mode=True),
        json=payload,
        timeout=60,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"TTS API error {response.status_code}: {response.text}")

    data = response.json()
    audios = data.get("audios", [])
    if not audios:
        return None

    return base64.b64decode(audios[0])


def sarvam_stt(audio_file: Any) -> str:
    require_key()

    audio_bytes = audio_file.getvalue()

    files = {
        "file": ("borrower_voice.wav", audio_bytes, "audio/wav"),
    }

    data = {
        "model": "saaras:v3",
        "mode": "codemix",
    }

    response = requests.post(
        STT_URL,
        headers=sarvam_headers(json_mode=False),
        files=files,
        data=data,
        timeout=60,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"STT API error {response.status_code}: {response.text}")

    result = response.json()
    return (
        result.get("transcript")
        or result.get("text")
        or result.get("transcription")
        or ""
    )


# =========================
# Business Logic
# =========================

def build_system_prompt(account: dict[str, Any], language: str) -> str:
    return f"""
You are a compliant multilingual loan collections voice agent for an Indian NBFC.

Borrower profile:
- Name: {account["name"]}
- City: {account["city"]}
- Loan product: {account["product"]}
- Overdue EMI: INR {account["emi_amount"]}
- Overdue days: {account["overdue_days"]}
- Risk band: {account["risk"]}
- Previous collection attempts: {account["attempts"]}

Business objective:
1. Remind borrower about overdue EMI.
2. Try to collect payment today.
3. If not possible, capture a clear promise-to-pay date.
4. If borrower cannot pay full amount, offer a practical split-payment plan.
5. If borrower disputes, refuses, threatens legal action, mentions fraud, harassment, death, hospitalization, or severe distress, escalate to a human agent.

Language instruction:
Respond in {language}.
If language is Hinglish, use natural Hindi-English code-mixing.

Compliance rules:
- Be polite and concise.
- Never threaten, shame, or pressure aggressively.
- Do not claim legal action.
- Do not make false promises.
- Ask for only one clear next step.
- Keep every reply under 80 words.
"""


def greeting(account: dict[str, Any], language: str) -> str:
    if language == "English":
        return (
            f"Hello {account['name']}, this is an automated reminder from your lender. "
            f"Your EMI of INR {account['emi_amount']} is {account['overdue_days']} days overdue. "
            f"Can you make the payment today or confirm a payment date?"
        )

    if language == "Tamil":
        return (
            f"Vanakkam {account['name']}. Ungal INR {account['emi_amount']} EMI "
            f"{account['overdue_days']} naal overdue. Indru payment panna mudiyuma, "
            f"illai payment date confirm pannalama?"
        )

    if language == "Kannada":
        return (
            f"Namaskara {account['name']}. Nimma INR {account['emi_amount']} EMI "
            f"{account['overdue_days']} dina overdue ide. Ivattu payment madabahuda, "
            f"athava payment date confirm madabahuda?"
        )

    if language == "Marathi":
        return (
            f"Namaskar {account['name']}. Tumchi INR {account['emi_amount']} EMI "
            f"{account['overdue_days']} divas overdue aahe. Aaj payment karu shakta ka, "
            f"ki payment date confirm karal?"
        )

    return (
        f"Namaste {account['name']}, aapki ₹{account['emi_amount']} EMI "
        f"{account['overdue_days']} din se overdue hai. Kya aap aaj payment kar sakte hain, "
        f"ya payment ke liye koi date confirm karna chahenge?"
    )


def extract_date_hint(text: str) -> str:
    text = text.lower()

    date_hints = [
        "today",
        "tomorrow",
        "friday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "saturday",
        "sunday",
        "kal",
        "parso",
        "10th",
        "15th",
        "month end",
        "salary",
    ]

    for hint in date_hints:
        if hint in text:
            return hint

    return "Customer committed date"


def classify_outcome(user_text: str, agent_reply: str) -> dict[str, Any]:
    text = f"{user_text} {agent_reply}".lower()

    refusal_terms = [
        "fraud",
        "legal",
        "complaint",
        "harassment",
        "wrong loan",
        "galat",
        "nahi karunga",
        "not pay",
        "won't pay",
        "refuse",
        "dispute",
        "death",
        "hospital",
    ]

    restructure_terms = [
        "installment",
        "part payment",
        "partial",
        "plan",
        "extension",
        "cannot pay full",
        "time chahiye",
        "split",
        "emi reduce",
    ]

    ptp_terms = [
        "friday",
        "monday",
        "tomorrow",
        "kal",
        "salary",
        "pay",
        "payment",
        "de dunga",
        "kar dunga",
        "will pay",
        "10th",
        "15th",
    ]

    if any(term in text for term in refusal_terms):
        return {
            "outcome": "Escalation",
            "risk_score": 90,
            "promise_date": "",
            "next_action": "Human collections manager callback",
        }

    if any(term in text for term in restructure_terms):
        return {
            "outcome": "Restructure Requested",
            "risk_score": 75,
            "promise_date": "To be confirmed",
            "next_action": "Offer split-payment plan",
        }

    if any(term in text for term in ptp_terms):
        return {
            "outcome": "Promise-to-Pay",
            "risk_score": 45,
            "promise_date": extract_date_hint(text),
            "next_action": "Reminder scheduled",
        }

    return {
        "outcome": "Follow-up Needed",
        "risk_score": 60,
        "promise_date": "",
        "next_action": "Continue conversation",
    }


def execute_workflow(
    account_id: str,
    account: dict[str, Any],
    user_text: str,
    agent_reply: str,
) -> None:
    outcome = classify_outcome(user_text, agent_reply)

    event = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "borrower": account_id,
        "customer": account["name"],
        "outcome": outcome["outcome"],
        "risk_score": outcome["risk_score"],
        "next_action": outcome["next_action"],
    }

    st.session_state.crm.append(event)

    if outcome["outcome"] == "Promise-to-Pay":
        st.session_state.reminders.append(
            {
                "borrower": account_id,
                "customer": account["name"],
                "channel": "WhatsApp/SMS",
                "promise_date": outcome["promise_date"],
                "message": f"Reminder queued for {account['name']}",
            }
        )
        st.session_state.analytics["Promise-to-Pay"] += 1

    elif outcome["outcome"] == "Escalation":
        st.session_state.escalations.append(
            {
                "borrower": account_id,
                "customer": account["name"],
                "reason": "High-risk refusal/dispute/distress",
                "owner": "Human collections manager",
            }
        )
        st.session_state.analytics["Escalations"] += 1

    elif outcome["outcome"] == "Restructure Requested":
        st.session_state.analytics["Restructure Requests"] += 1


def play_tts(text: str, language: str) -> None:
    try:
        audio = sarvam_tts(text, language)
        if audio:
            st.audio(audio, format="audio/wav")
    except Exception as exc:
        st.error(f"TTS failed: {exc}")


def process_customer_message(
    account_id: str,
    account: dict[str, Any],
    language: str,
    user_text: str,
    voice_enabled: bool,
) -> None:
    if not user_text.strip():
        return

    st.session_state.messages.append({"role": "user", "content": user_text})

    model_messages = [{"role": "system", "content": build_system_prompt(account, language)}]
    model_messages.extend(st.session_state.messages[-10:])

    with st.spinner("Sarvam 105B is reasoning..."):
        reply = sarvam_chat(model_messages)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    execute_workflow(account_id, account, user_text, reply)

    if voice_enabled:
        play_tts(reply, language)

    st.rerun()


# =========================
# UI
# =========================

st.title("Sarvam AI Collections & Recovery Agent")
st.caption("Voice-enabled multilingual EMI reminder and recovery workflow for Indian NBFCs and banks")

with st.sidebar:
    st.header("Demo Controls")

    account_id = st.selectbox("Borrower", list(ACCOUNTS.keys()))
    language = st.selectbox(
        "Conversation language",
        ["Hindi", "English", "Hinglish", "Tamil", "Kannada", "Marathi"],
    )

    voice_enabled = st.toggle("Play agent replies using Sarvam TTS", value=True)

    if SARVAM_API_KEY:
        st.success("Live Sarvam API mode enabled")
    else:
        st.error("SARVAM_API_KEY missing")

    if st.button("Reset Call"):
        reset_call()
        st.rerun()

account = ACCOUNTS[account_id]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Borrower", account["name"])
m2.metric("Overdue EMI", f"₹{account['emi_amount']:,}")
m3.metric("Overdue Days", account["overdue_days"])
m4.metric("Risk", account["risk"])

st.divider()

left, right = st.columns([1.35, 1])

with left:
    st.subheader("Conversation")

    start_col, sample_col = st.columns([1, 2])

    with start_col:
        if st.button("Start Call", use_container_width=True):
            st.session_state.analytics["Calls"] += 1
            first_message = greeting(account, language)
            st.session_state.messages.append(
                {"role": "assistant", "content": first_message}
            )
            if voice_enabled:
                play_tts(first_message, language)
            st.rerun()

    with sample_col:
        st.caption("Use voice response below, or type the borrower response.")

    for msg in st.session_state.messages:
        role = msg["role"]
        label = "Agent" if role == "assistant" else "Borrower"
        with st.chat_message("assistant" if role == "assistant" else "user"):
            st.write(f"**{label}:** {msg['content']}")

    st.markdown("### Borrower Voice Response")

    voice_audio = st.audio_input("Record borrower response")

    if voice_audio is not None:
        if st.button("Transcribe and Send Voice Response", use_container_width=True):
            try:
                with st.spinner("Sarvam Saaras v3 is transcribing voice..."):
                    transcript = sarvam_stt(voice_audio)

                st.session_state.last_transcript = transcript
                st.success(f"Transcribed: {transcript}")

                process_customer_message(
                    account_id=account_id,
                    account=account,
                    language=language,
                    user_text=transcript,
                    voice_enabled=voice_enabled,
                )
            except Exception as exc:
                st.error(f"STT failed: {exc}")

    if st.session_state.last_transcript:
        st.info(f"Last STT transcript: {st.session_state.last_transcript}")

    typed_text = st.chat_input(
        "Or type borrower response, e.g. Salary kal aa jayegi, Friday ko payment kar dunga"
    )

    if typed_text:
        process_customer_message(
            account_id=account_id,
            account=account,
            language=language,
            user_text=typed_text,
            voice_enabled=voice_enabled,
        )

with right:
    st.subheader("Borrower Context")
    st.write(f"**Name:** {account['name']}")
    st.write(f"**City:** {account['city']}")
    st.write(f"**Loan Product:** {account['product']}")
    st.write(f"**EMI Overdue:** ₹{account['emi_amount']:,}")
    st.write(f"**Days Overdue:** {account['overdue_days']}")
    st.write(f"**Previous Attempts:** {account['attempts']}")

    st.subheader("Agent Objective")
    st.write(
        "Collect payment today, capture a promise-to-pay date, offer restructuring if needed, "
        "or escalate high-risk cases to a human collections manager."
    )

    st.subheader("Live Analytics")
    st.dataframe(
        pd.DataFrame([st.session_state.analytics]),
        use_container_width=True,
        hide_index=True,
    )

    latest_risk = st.session_state.crm[-1]["risk_score"] if st.session_state.crm else 50
    st.progress(latest_risk / 100, text=f"Current Risk Score: {latest_risk}/100")

st.divider()

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("CRM Events")
    if st.session_state.crm:
        st.dataframe(pd.DataFrame(st.session_state.crm), use_container_width=True, hide_index=True)
    else:
        st.info("No CRM events yet.")

with c2:
    st.subheader("Reminder Queue")
    if st.session_state.reminders:
        st.dataframe(pd.DataFrame(st.session_state.reminders), use_container_width=True, hide_index=True)
    else:
        st.info("No reminders queued yet.")

with c3:
    st.subheader("Escalation Queue")
    if st.session_state.escalations:
        st.dataframe(pd.DataFrame(st.session_state.escalations), use_container_width=True, hide_index=True)
    else:
        st.info("No escalations yet.")

st.divider()

st.subheader("Post-call Summary")

if st.button("Generate Summary"):
    if not st.session_state.messages:
        st.warning("Start a call first.")
    else:
        transcript = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
        )

        summary_messages = [
            {
                "role": "system",
                "content": (
                    "Create a concise collections call summary with: disposition, "
                    "promise date, risk score, next action, and compliance notes."
                ),
            },
            {"role": "user", "content": transcript},
        ]

        try:
            with st.spinner("Generating summary with Sarvam 105B..."):
                summary = sarvam_chat(summary_messages)
            st.write(summary)
        except Exception as exc:
            st.error(f"Summary failed: {exc}")
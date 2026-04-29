"""Pitwall-AI Streamlit frontend.

Layout (two-panel):
┌──────────────────────────────────────────────────────────┐
│  PITWALL-AI         [Select Race ▼]  [Lap Slider: 34]    │
├───────────────────────────┬──────────────────────────────┤
│  RACE STANDINGS           │  RACE SUMMARY + WEATHER      │
│  (live table with tire    │  (auto-updated on lap change) │
│   compounds + gaps)       │                              │
├───────────────────────────┴──────────────────────────────┤
│  STRATEGY CHAT                                           │
│  (natural language Q&A via /api/chat)                    │
└──────────────────────────────────────────────────────────┘

Run with:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000/api"

COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFD700",
    "HARD": "#EEEEEE",
    "INTERMEDIATE": "#00C853",
    "WET": "#2196F3",
    "UNKNOWN": "#888888",
}

COMPOUND_EMOJI = {
    "SOFT": "S",
    "MEDIUM": "M",
    "HARD": "H",
    "INTERMEDIATE": "I",
    "WET": "W",
    "UNKNOWN": "?",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pitwall-AI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)  # Cache race list for 5 minutes
def fetch_races() -> list[dict]:
    try:
        r = requests.get(f"{API_BASE}/races", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Could not load races: {exc}")
        return []


@st.cache_data(ttl=60)  # Cache per-lap state for 1 minute
def fetch_race_state(session_key: int, lap: int) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/races/{session_key}/lap/{lap}", timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Could not load race state: {exc}")
        return None


@st.cache_data(ttl=120)
def fetch_race_summary(session_key: int, lap: int) -> str:
    try:
        r = requests.get(f"{API_BASE}/races/{session_key}/summary/{lap}", timeout=30)
        r.raise_for_status()
        return r.json().get("summary", "Summary unavailable")
    except Exception:
        return "Summary unavailable — check API connection."


def stream_chat(
    session_key: int,
    lap: int,
    message: str,
    history: list[dict],
):
    """Stream chat response via SSE. Yields (event_type, data) tuples.

    Event types: "meta", "token", "done", "error".
    """
    try:
        r = requests.post(
            f"{API_BASE}/chat/stream",
            json={
                "session_key": session_key,
                "current_lap": lap,
                "message": message,
                "conversation_history": history,
            },
            timeout=90,
            stream=True,
        )
        r.raise_for_status()

        current_event = "token"
        for line in r.iter_lines(decode_unicode=True):
            if line is None:
                continue
            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
                yield current_event, data
            # blank lines separate events — reset to default
            elif line == "":
                current_event = "token"

    except Exception as exc:
        yield "error", str(exc)


def _compound_badge(compound: str) -> str:
    letter = COMPOUND_EMOJI.get(compound, "?")
    color = COMPOUND_COLORS.get(compound, "#888")
    return f'<span style="background:{color};color:#000;padding:2px 7px;border-radius:4px;font-weight:bold;font-size:12px">{letter}</span>'


# ---------------------------------------------------------------------------
# Initialise session state
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

if "selected_session" not in st.session_state:
    st.session_state.selected_session: int | None = None

if "selected_lap" not in st.session_state:
    st.session_state.selected_lap: int = 1

# ---------------------------------------------------------------------------
# Header row
# ---------------------------------------------------------------------------

header_col1, header_col2, header_col3 = st.columns([2, 3, 2])

with header_col1:
    st.markdown("## 🏎️ Pitwall-AI")
    st.caption("AI-powered F1 race strategy · ask me anything")

# Load races for the selector
races = fetch_races()
race_options: dict[str, int] = {}
for race in races:
    circuit = race.get("circuit_short_name") or race.get("location") or "Unknown"
    country = race.get("country_name") or race.get("country_code") or "?"
    label = f"{race.get('year', '?')} — {circuit} ({country})"
    race_options[label] = race.get("session_key", 0)

with header_col2:
    if race_options:
        selected_label = st.selectbox(
            "Select Race",
            options=list(race_options.keys()),
            key="race_select",
            label_visibility="collapsed",
        )
        st.session_state.selected_session = race_options[selected_label]
    else:
        st.warning("No races available — is the API running?")
        st.stop()

# Determine max lap for slider (fetch full state first if needed)
max_laps = 60  # sensible default before we know the real total
if st.session_state.selected_session:
    full_state_preview = fetch_race_state(st.session_state.selected_session, 9999)
    if full_state_preview:
        max_laps = full_state_preview.get("total_laps", 60) or 60

with header_col3:
    st.session_state.selected_lap = st.slider(
        "Lap",
        min_value=1,
        max_value=max_laps,
        value=min(st.session_state.selected_lap, max_laps),
        key="lap_slider",
    )

st.divider()

# ---------------------------------------------------------------------------
# Main panel — Standings | Summary + Weather
# ---------------------------------------------------------------------------

session_key = st.session_state.selected_session
current_lap = st.session_state.selected_lap

left_col, right_col = st.columns([1.2, 1])

race_state = fetch_race_state(session_key, current_lap) if session_key else None

with left_col:
    st.markdown("### Race Standings")

    if race_state and race_state.get("drivers"):
        drivers = sorted(race_state["drivers"], key=lambda d: d.get("position", 99))

        # Build table data
        rows_html = []
        for d in drivers:
            pos = d.get("position", "-")
            num = d.get("driver_number", "-")
            name = d.get("name", "Unknown")
            team = d.get("team", "")
            compound = d.get("tire_compound", "UNKNOWN")
            stint_len = d.get("stint_length", 0)
            gap = d.get("gap_to_leader")
            gap_str = f"+{gap:.3f}s" if gap and gap > 0 else "LEADER"
            last_lap = d.get("last_lap_time")
            last_lap_str = f"{last_lap:.3f}s" if last_lap else "—"
            badge = _compound_badge(compound)
            retired = "💀" if d.get("is_retired") else ""
            pit = "🔧" if d.get("is_in_pit") else ""

            rows_html.append(
                f"<tr>"
                f"<td style='text-align:center;padding:4px 8px'><b>P{pos}</b></td>"
                f"<td style='padding:4px 8px'>{badge} #{num} {retired}{pit}</td>"
                f"<td style='padding:4px 8px'><b>{name}</b><br><small style='color:#888'>{team}</small></td>"
                f"<td style='padding:4px 8px;text-align:right'>{gap_str}</td>"
                f"<td style='padding:4px 8px;text-align:right'>{last_lap_str}</td>"
                f"<td style='padding:4px 8px;text-align:center'>Lap {stint_len}</td>"
                f"</tr>"
            )

        table_html = f"""
        <table style='width:100%;border-collapse:collapse;font-size:13px'>
          <thead>
            <tr style='border-bottom:1px solid #444;color:#888;font-size:11px'>
              <th style='padding:4px 8px'>Pos</th>
              <th style='padding:4px 8px'>Tire #</th>
              <th style='padding:4px 8px'>Driver</th>
              <th style='padding:4px 8px;text-align:right'>Gap</th>
              <th style='padding:4px 8px;text-align:right'>Last Lap</th>
              <th style='padding:4px 8px;text-align:center'>Stint</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("No driver data — check API connection or select a race.")

with right_col:
    # Race Summary
    st.markdown("### Race Narrative")
    with st.spinner("Generating summary..."):
        summary = fetch_race_summary(session_key, current_lap) if session_key else "Select a race."
    st.markdown(
        f'<div style="background:#1a1a2e;padding:14px;border-radius:8px;font-size:13px;line-height:1.6">{summary}</div>',
        unsafe_allow_html=True,
    )

    # Weather widget
    if race_state and race_state.get("weather"):
        w = race_state["weather"]
        st.markdown("### Weather")
        w_col1, w_col2, w_col3 = st.columns(3)
        with w_col1:
            air = w.get("air_temp")
            st.metric("Air Temp", f"{air:.0f}°C" if air else "—")
        with w_col2:
            track = w.get("track_temp")
            st.metric("Track Temp", f"{track:.0f}°C" if track else "—")
        with w_col3:
            rain = w.get("rainfall", False)
            st.metric("Rain", "YES ☔" if rain else "No ☀️")
        hum = w.get("humidity")
        if hum:
            st.progress(int(hum), text=f"Humidity: {hum:.0f}%")

st.divider()

# ---------------------------------------------------------------------------
# Strategy Chat
# ---------------------------------------------------------------------------

st.markdown("### Strategy Chat")
st.caption("Ask me anything — 'Should Verstappen pit now?', 'Compare Norris vs Leclerc tires', 'What happened at this track in 2024?'")

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])

# Chat input
user_input = st.chat_input("Ask a strategy question...")
if user_input and session_key:
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build conversation history for the API (last 10 messages, alternating)
    api_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_history[-10:]
        if m["role"] in ("user", "assistant")
    ]

    # Stream the response via SSE
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.caption("⏳ Running agent pipeline...")

        response_placeholder = st.empty()
        meta_placeholder = st.empty()
        strategy_container = st.container()

        response_text = ""
        agents: list[str] = []
        strategy: dict | None = None
        total_ms: float = 0
        pipeline_ms: float = 0
        t0 = time.monotonic()

        for event_type, data in stream_chat(
            session_key=session_key,
            lap=current_lap,
            message=user_input,
            history=api_history[:-1],
        ):
            if event_type == "meta":
                meta_info = json.loads(data)
                agents = meta_info.get("agents_used", [])
                strategy = meta_info.get("strategy_data")
                pipeline_ms = meta_info.get("pipeline_time_ms", 0)
                status_placeholder.empty()

            elif event_type == "token":
                response_text += data
                response_placeholder.markdown(response_text + "▌")

            elif event_type == "done":
                done_info = json.loads(data)
                total_ms = done_info.get("total_time_ms", round((time.monotonic() - t0) * 1000))

            elif event_type == "error":
                response_text = f"Error: {data}"

        # Final render (remove cursor)
        response_placeholder.markdown(response_text or "No response")

        if not total_ms:
            total_ms = round((time.monotonic() - t0) * 1000)

        meta = f"Agents: {' → '.join(agents)} | {total_ms:.0f}ms"
        meta_placeholder.caption(meta)

        # Strategy card
        if strategy:
            with strategy_container:
                action = strategy.get("recommended_action", "")
                compound = strategy.get("recommended_compound", "")
                confidence = strategy.get("confidence", 0)
                pit_window = strategy.get("optimal_pit_window")

                with st.expander("📊 Strategy Details", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Recommendation", action)
                    c2.metric("Compound", compound or "TBD")
                    c3.metric("Confidence", f"{confidence:.0%}")
                    if pit_window:
                        st.info(f"Pit window: Lap {pit_window[0]}–{pit_window[1]}")
                    undercut = strategy.get("undercut_viable")
                    overcut = strategy.get("overcut_viable")
                    if undercut or overcut:
                        flags = []
                        if undercut:
                            flags.append("✅ Undercut viable")
                        if overcut:
                            flags.append("✅ Overcut viable")
                        st.success(" | ".join(flags))

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response_text, "meta": meta}
    )

elif user_input and not session_key:
    st.warning("Please select a race first.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Data: [OpenF1 API](https://openf1.org) · "
    "LLM: Groq Llama 3.3 70B + Google Gemini fallback · "
    "Built with LangGraph, FastAPI, Streamlit"
)

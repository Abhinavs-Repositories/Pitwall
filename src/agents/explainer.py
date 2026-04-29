"""Explainer Agent — converts structured strategy data into natural language.

System prompt makes responses sound like a real F1 race engineer / commentator.
References specific data points: lap times, gaps, degradation rates, precedents.
Supports streaming (Groq natively supports it for better UX).
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.state import AgentState
from src.core.llm import get_llm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert F1 race strategist working as a virtual pit wall engineer.
Your job is to explain race strategy recommendations clearly and concisely.

Guidelines:
- Reference specific data points: lap times, tire ages, gaps, degradation rates
- Sound like a real F1 race engineer — confident, data-driven, precise
- Be concise but thorough (3-6 sentences for most answers)
- When historical precedent is available, reference it naturally
- Mention confidence levels when expressing uncertainty
- Use F1 terminology naturally (undercut, overcut, pit window, tire cliff, stint, DRS)
- If the user asks a simple status question, keep it brief
- Always end strategy recommendations with a clear action
- Pit stop data: "Stops: Lap X (COMPOUND_BEFORE→COMPOUND_AFTER)" means the driver pitted on that lap.
  "No pit stops yet" means the driver has NOT pitted. "stint lap N" means laps on current tires, NOT the lap they pitted.
- Only state facts that are explicitly present in the data. Do NOT infer or guess pit stop information."""

_OFF_TOPIC_RESPONSE = (
    "I'm focused on F1 race strategy and analysis. "
    "I can help with pit stop timing, tire degradation, weather impact, "
    "race standings, and strategic comparisons between drivers. "
    "What would you like to know about the race?"
)


def explainer_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: produce the final natural language response."""
    # Short-circuit for off-topic queries — no LLM call needed
    if state.query_type == "off_topic":
        return {"final_response": _OFF_TOPIC_RESPONSE, "agents_used": ["explainer"]}

    # Check for upstream errors or missing data
    if state.errors and state.race_state is None:
        error_summary = "; ".join(state.errors[:3])
        return {
            "final_response": f"I encountered issues fetching race data: {error_summary}. "
                              "Please check the session key and try again.",
            "agents_used": ["explainer"],
        }

    context = _build_context(state)
    prompt = _build_prompt(state.user_message, context, state.query_type)

    try:
        llm = get_llm(streaming=False)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *state.conversation_history,
            {"role": "user", "content": prompt},
        ]
        response = llm.invoke(messages)
        final_response = str(response.content).strip()
    except Exception as exc:
        logger.error("explainer_node LLM call failed: %s", exc)
        final_response = _fallback_response(state)

    return {"final_response": final_response, "agents_used": ["explainer"]}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Which context sections each query_type needs
_CONTEXT_SECTIONS: dict[str, set[str]] = {
    "race_status":   {"race_summary", "standings", "weather_current"},
    "tire_analysis": {"race_summary", "standings", "weather_current"},
    "weather":       {"race_summary", "weather_current", "weather_alert"},
    "strategy":      {"race_summary", "standings", "weather_current", "weather_alert", "strategy", "historical", "track"},
    "comparison":    {"race_summary", "standings", "strategy"},
    "historical":    {"race_summary", "historical", "track"},
    "off_topic":     set(),
}


def _build_context(state: AgentState) -> str:
    """Build a structured context block for the LLM prompt, filtered by query_type."""
    sections = _CONTEXT_SECTIONS.get(state.query_type, _CONTEXT_SECTIONS["strategy"])
    parts: list[str] = []

    # Race state summary
    if state.race_state and "race_summary" in sections:
        rs = state.race_state
        parts.append(
            f"Race: {rs.meeting_name} — Lap {rs.current_lap}/{rs.total_laps}\n"
            f"Status: {rs.session_status}"
        )

        # Full grid standings
        if "standings" in sections:
            sorted_drivers = sorted(rs.drivers, key=lambda d: d.position)
            standings_lines = []
            for d in sorted_drivers:
                gap = f"+{d.gap_to_leader:.3f}s" if d.gap_to_leader and d.gap_to_leader > 0 else "LEADER"
                deg_key = str(d.driver_number)
                deg_info = ""
                if deg_key in state.tire_degradations:
                    deg = state.tire_degradations[deg_key]
                    deg_info = f" | Deg: {deg.deg_rate_per_lap:.3f}s/lap"
                    if deg.laps_remaining_estimate is not None:
                        deg_info += f" | ~{deg.laps_remaining_estimate} laps to cliff"
                # Pit stop summary
                pit_info = ""
                if d.pit_stops:
                    stops = ", ".join(
                        f"Lap {ps.lap_number} ({ps.compound_before.value if ps.compound_before else '?'}"
                        f"→{ps.compound_after.value if ps.compound_after else '?'})"
                        for ps in d.pit_stops
                    )
                    pit_info = f" | Stops: {stops}"
                else:
                    pit_info = " | No pit stops yet"

                # Full stint history
                stint_info = ""
                if len(d.stints) > 1:
                    stint_parts = [
                        f"S{s.stint_number}:{s.compound.value} L{s.lap_start}-{s.lap_end}"
                        for s in sorted(d.stints, key=lambda s: s.stint_number)
                    ]
                    stint_info = f" | Stints: {' → '.join(stint_parts)}"

                standings_lines.append(
                    f"  P{d.position} #{d.driver_number} {d.name} ({d.team}) "
                    f"| Current: {d.tire_compound.value} (stint lap {d.stint_length}) | {gap}{deg_info}"
                    f"{pit_info}{stint_info}"
                )
            parts.append("Current standings:\n" + "\n".join(standings_lines))

        # Weather
        if "weather_current" in sections:
            w = rs.weather
            if w.air_temp is not None:
                rain_str = "YES" if w.rainfall else "No"
                parts.append(
                    f"Weather: Air {w.air_temp}°C | Track {w.track_temp}°C | "
                    f"Humidity {w.humidity}% | Rain: {rain_str}"
                )

    # Weather alert
    if state.weather_alert and "weather_alert" in sections:
        parts.append(f"Weather alert: {state.weather_alert}")

    # Strategy recommendations
    if state.strategy_recommendations and "strategy" in sections:
        rec_lines = []
        for driver_num_str, rec in state.strategy_recommendations.items():
            driver_name = _driver_name(state, int(driver_num_str))
            rec_lines.append(
                f"  {driver_name}: {rec.recommended_action} "
                f"(confidence {rec.confidence:.0%}) | {rec.reasoning[:120]}"
            )
        parts.append("Strategy recommendations:\n" + "\n".join(rec_lines))

    # Historical context
    if state.historical_context and "historical" in sections:
        hist_lines = []
        for h in state.historical_context[:2]:
            hist_lines.append(
                f"  {h.race_name} {h.year}: {h.winner} won with {h.winner_strategy} strategy. {h.summary[:100]}"
            )
        parts.append("Historical precedents:\n" + "\n".join(hist_lines))

    # Track characteristics
    if state.track_characteristics and "track" in sections:
        tc = state.track_characteristics
        parts.append(
            f"Track data: typical={tc.get('typical_strategy', '?')} | "
            f"pit loss={tc.get('pit_loss_seconds', '?')}s | "
            f"SC prob={tc.get('safety_car_probability', '?')}"
        )

    return "\n\n".join(parts)


def _build_prompt(user_message: str, context: str, query_type: str) -> str:
    return (
        f"[Context]\n{context}\n\n"
        f"[Query type: {query_type}]\n\n"
        f"[User question]\n{user_message}\n\n"
        f"Please answer the user's question using the context above. "
        f"Be specific, reference the data, and give a clear recommendation if applicable."
    )


def _driver_name(state: AgentState, driver_number: int) -> str:
    if state.race_state:
        for d in state.race_state.drivers:
            if d.driver_number == driver_number:
                return d.name
    return f"Driver #{driver_number}"


def _fallback_response(state: AgentState) -> str:
    """Minimal plain-text response when LLM is unavailable."""
    if state.strategy_recommendations:
        recs = []
        for dn_str, rec in state.strategy_recommendations.items():
            name = _driver_name(state, int(dn_str))
            recs.append(f"{name}: {rec.recommended_action} — {rec.reasoning[:100]}")
        return "Strategy analysis:\n" + "\n".join(recs)
    if state.race_state:
        rs = state.race_state
        return f"Lap {rs.current_lap}/{rs.total_laps} — {len(rs.drivers)} drivers active."
    return "Analysis unavailable — please check API keys and try again."

"""Weather Agent — monitors conditions and flags rain threats.

Fetches weather history for the session, calls the pure-Python weather
analyser, and writes weather_history + weather_alert to shared state.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.state import AgentState
from src.analysis.weather import WeatherEvent, detect_weather_changes
from src.data.models import WeatherState
from src.data.openf1_client import OpenF1Client

logger = logging.getLogger(__name__)


async def weather_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: fetch weather history and detect changes."""
    session_key = state.session_key
    if not session_key:
        return {"errors": state.errors + ["weather_node: no session_key"]}

    try:
        async with OpenF1Client() as client:
            raw_weather = await client.get_weather(session_key=session_key)
    except Exception as exc:
        logger.error("weather_node fetch failed: %s", exc)
        return {"errors": state.errors + [f"Weather fetch error: {exc}"]}

    if not raw_weather:
        return {"weather_history": [], "weather_alert": "", "agents_used": list(state.agents_used) + ["weather"]}

    # Convert raw dicts → WeatherState list
    weather_history: list[WeatherState] = []
    for w in raw_weather:
        try:
            weather_history.append(
                WeatherState(
                    air_temp=w.get("air_temperature"),
                    track_temp=w.get("track_temperature"),
                    humidity=w.get("humidity"),
                    rainfall=bool(w.get("rainfall", False)),
                    wind_speed=w.get("wind_speed"),
                    wind_direction=w.get("wind_direction"),
                )
            )
        except Exception:
            continue

    # Optionally filter to up_to_lap if we had lap timestamps (best effort)
    events = detect_weather_changes(weather_history)

    alert = ""
    if any(e.event == WeatherEvent.RAIN_THREAT for e in events):
        alert = "Rain threat detected — humidity rising and/or temperature dropping. Monitor for wet conditions."
    elif any(e.event == WeatherEvent.RAIN_STARTED for e in events):
        alert = "Rain has started — consider switching to intermediate or wet tires."
    elif any(e.event == WeatherEvent.RAIN_STOPPED for e in events):
        alert = "Rain has stopped — track drying. Slick tires may soon be viable."

    agents_used = list(state.agents_used) + ["weather"]

    return {
        "weather_history": weather_history,
        "weather_alert": alert,
        "agents_used": agents_used,
    }

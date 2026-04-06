"""Weather change detection for Pitwall-AI.

All logic is pure Python — no LLM, no external calls.

Detection rules (conservative — we'd rather flag a false positive
than miss an incoming rain event):
  - RAIN_THREAT:  humidity > 80 % AND track temp dropping > 3 °C over last N readings
  - RAIN_STARTED: rainfall flag flips from False → True
  - RAIN_STOPPED: rainfall flag flips from True → False
  - TEMP_SPIKE:   track temp rises > 5 °C (affects tire deg assumptions)
  - WIND_SHIFT:   wind speed jump > 10 km/h (affects aero balance)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.data.models import WeatherState

logger = logging.getLogger(__name__)

# Thresholds
RAIN_THREAT_HUMIDITY: float = 80.0          # %
RAIN_THREAT_TEMP_DROP: float = 3.0          # °C over window
TEMP_SPIKE_THRESHOLD: float = 5.0           # °C rise
WIND_SHIFT_THRESHOLD: float = 10.0          # km/h jump


class WeatherEvent(str, Enum):
    RAIN_THREAT = "RAIN_THREAT"
    RAIN_STARTED = "RAIN_STARTED"
    RAIN_STOPPED = "RAIN_STOPPED"
    TEMP_SPIKE = "TEMP_SPIKE"
    WIND_SHIFT = "WIND_SHIFT"


@dataclass(frozen=True)
class WeatherChange:
    event: WeatherEvent
    description: str
    severity: str  # "LOW" | "MEDIUM" | "HIGH"
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_weather_changes(
    history: list[WeatherState],
) -> list[WeatherChange]:
    """Scan a time-ordered list of WeatherState snapshots and return detected events.

    Args:
        history: Ordered list (oldest → newest) of WeatherState readings.
                 Typically one per lap or one per minute.
    """
    if len(history) < 2:
        return []

    events: list[WeatherChange] = []

    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]
        events.extend(_compare(prev, curr, history[: i + 1]))

    return events


def is_rain_threat(weather_history: list[WeatherState]) -> bool:
    """Return True if conditions suggest rain is incoming."""
    if not weather_history:
        return False
    changes = detect_weather_changes(weather_history)
    return any(c.event == WeatherEvent.RAIN_THREAT for c in changes)


def get_current_conditions_summary(weather: WeatherState) -> str:
    """Return a one-line human-readable weather summary."""
    parts: list[str] = []
    if weather.air_temp is not None:
        parts.append(f"Air {weather.air_temp:.1f}°C")
    if weather.track_temp is not None:
        parts.append(f"Track {weather.track_temp:.1f}°C")
    if weather.humidity is not None:
        parts.append(f"Humidity {weather.humidity:.0f}%")
    if weather.rainfall:
        parts.append("RAIN")
    elif weather.humidity is not None and weather.humidity > RAIN_THREAT_HUMIDITY:
        parts.append("High humidity — rain risk")
    if weather.wind_speed is not None:
        parts.append(f"Wind {weather.wind_speed:.1f} km/h")
    return " | ".join(parts) if parts else "No weather data"


def recommend_tire_for_conditions(weather: WeatherState, laps_remaining: int) -> str:
    """Return a simple compound recommendation based purely on weather.

    This feeds into the Strategy agent as one input — not a standalone decision.
    """
    if weather.rainfall:
        if laps_remaining > 15:
            return "INTERMEDIATE"  # swap to inters immediately
        return "WET"  # full wet if heavy rain expected

    # Dry conditions — guidance only, strategy agent makes final call
    if weather.track_temp is not None and weather.track_temp > 45:
        return "HARD"  # high track temp eats softer compounds

    if laps_remaining <= 10:
        return "SOFT"
    if laps_remaining <= 22:
        return "MEDIUM"
    return "HARD"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compare(
    prev: WeatherState,
    curr: WeatherState,
    history_so_far: list[WeatherState],
) -> list[WeatherChange]:
    events: list[WeatherChange] = []

    # Rain started
    if not prev.rainfall and curr.rainfall:
        events.append(
            WeatherChange(
                event=WeatherEvent.RAIN_STARTED,
                description="Rain has started falling on track",
                severity="HIGH",
                details={"humidity": curr.humidity, "track_temp": curr.track_temp},
            )
        )
        logger.warning("Weather: rain started")

    # Rain stopped
    elif prev.rainfall and not curr.rainfall:
        events.append(
            WeatherChange(
                event=WeatherEvent.RAIN_STOPPED,
                description="Rain has stopped — track drying",
                severity="MEDIUM",
                details={"humidity": curr.humidity, "track_temp": curr.track_temp},
            )
        )
        logger.info("Weather: rain stopped")

    # Rain threat — no rain yet but conditions converging
    elif not curr.rainfall:
        threat = _check_rain_threat(history_so_far)
        if threat:
            events.append(threat)

    # Track temperature spike (affects tire behaviour assumptions)
    if prev.track_temp is not None and curr.track_temp is not None:
        delta = curr.track_temp - prev.track_temp
        if delta >= TEMP_SPIKE_THRESHOLD:
            events.append(
                WeatherChange(
                    event=WeatherEvent.TEMP_SPIKE,
                    description=f"Track temp rose {delta:.1f}°C — faster tire degradation likely",
                    severity="MEDIUM",
                    details={"prev_temp": prev.track_temp, "curr_temp": curr.track_temp, "delta": delta},
                )
            )

    # Significant wind shift
    if prev.wind_speed is not None and curr.wind_speed is not None:
        delta_wind = abs(curr.wind_speed - prev.wind_speed)
        if delta_wind >= WIND_SHIFT_THRESHOLD:
            events.append(
                WeatherChange(
                    event=WeatherEvent.WIND_SHIFT,
                    description=f"Wind speed changed by {delta_wind:.1f} km/h",
                    severity="LOW",
                    details={"prev_wind": prev.wind_speed, "curr_wind": curr.wind_speed},
                )
            )

    return events


def _check_rain_threat(history: list[WeatherState]) -> WeatherChange | None:
    """Detect converging conditions: high humidity + falling track temp."""
    if len(history) < 3:
        return None

    recent = history[-3:]  # last 3 readings

    humidities = [w.humidity for w in recent if w.humidity is not None]
    track_temps = [w.track_temp for w in recent if w.track_temp is not None]

    if not humidities or not track_temps:
        return None

    avg_humidity = sum(humidities) / len(humidities)
    if avg_humidity < RAIN_THREAT_HUMIDITY:
        return None

    # Check for consistent temp drop
    if len(track_temps) >= 2 and (track_temps[0] - track_temps[-1]) >= RAIN_THREAT_TEMP_DROP:
        return WeatherChange(
            event=WeatherEvent.RAIN_THREAT,
            description=(
                f"Rain threat: humidity {avg_humidity:.0f}% and "
                f"track temp dropping ({track_temps[0]:.1f}→{track_temps[-1]:.1f}°C)"
            ),
            severity="MEDIUM",
            details={
                "avg_humidity": avg_humidity,
                "track_temp_start": track_temps[0],
                "track_temp_end": track_temps[-1],
            },
        )

    return None

"""F1 pit strategy calculation — pure Python, zero LLM calls.

The Strategy Agent calls these functions and passes the results to the
LLM Explainer agent for natural language output.

Key concepts:
  - Pit loss:          Time lost in the pit lane (~22 s at most tracks).
  - Fresh tire gain:   Typical pace advantage of new tires vs old ones.
                       ~1–2 s/lap for the first 2–3 laps, tapering off.
  - Undercut window:   Gap to car ahead < pit_loss − fresh_tire_advantage.
                       i.e. the gap is small enough that pitting first, getting
                       clean air + fresh tires, and coming out ahead is viable.
  - Overcut window:    Car behind pits first; staying out on old rubber is
                       beneficial if track position + lap delta outweighs the
                       fresh-tire threat.
  - Optimal pit lap:   Balance tire cliff timing vs traffic vs race remaining.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.analysis.tire_deg import TireDegradation
from src.analysis.weather import WeatherEvent, detect_weather_changes
from src.data.models import (
    DriverState,
    RaceState,
    StrategyRecommendation,
    TireCompound,
    WeatherState,
)

logger = logging.getLogger(__name__)

# Track-level defaults (used when no per-track data is available)
DEFAULT_PIT_LOSS_S: float = 22.0          # seconds lost during a pit stop
FRESH_TIRE_GAIN_PER_LAP_S: float = 0.8   # extra pace per lap on fresh rubber (conservative)
FRESH_TIRE_ADVANTAGE_LAPS: int = 3        # laps where fresh-tire advantage is meaningful
SAFETY_CAR_PROB_DEFAULT: float = 0.15    # baseline SC probability per remaining stint


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PitWindow:
    earliest_lap: int
    latest_lap: int
    ideal_lap: int


def calculate_optimal_pit_window(
    driver: DriverState,
    deg: TireDegradation | None,
    race_state: RaceState,
    pit_loss: float = DEFAULT_PIT_LOSS_S,
) -> PitWindow | None:
    """Calculate the optimal lap window to pit.

    Logic:
    1. Base window anchored on predicted cliff lap (if available).
    2. Bounded by: earliest = current_lap + 1 (can't pit in the past),
                   latest   = total_laps - MIN_LAPS_ON_NEW_TIRES.
    3. Ideal lap = cliff_lap - 1 (pit just before cliff) or midpoint.

    Returns None if the driver has no remaining stint or is already pitted.
    """
    current_lap = race_state.current_lap
    total_laps = race_state.total_laps
    laps_remaining = total_laps - current_lap

    if laps_remaining <= 0:
        return None

    # Minimum laps we need on fresh tires to make a stop worthwhile
    min_laps_on_new_tires = _min_laps_for_compound(driver.tire_compound)

    latest_viable = total_laps - min_laps_on_new_tires
    earliest_viable = current_lap + 1

    if earliest_viable > latest_viable:
        return None  # no valid window — too close to end

    if deg and deg.predicted_cliff_lap:
        cliff = deg.predicted_cliff_lap
        # Pit 1 lap before cliff (pro-active) but respect window bounds
        ideal = max(earliest_viable, min(cliff - 1, latest_viable))
        earliest = max(earliest_viable, cliff - 3)
        latest = min(latest_viable, cliff + 2)
    else:
        # No cliff data — default to pitting at ~60–70 % of laps remaining
        offset = max(1, int(laps_remaining * 0.35))
        ideal = min(current_lap + offset, latest_viable)
        earliest = max(earliest_viable, ideal - 3)
        latest = min(latest_viable, ideal + 4)

    return PitWindow(earliest_lap=earliest, latest_lap=latest, ideal_lap=ideal)


def evaluate_undercut(
    driver: DriverState,
    car_ahead: DriverState,
    deg_driver: TireDegradation | None,
    deg_ahead: TireDegradation | None,
    pit_loss: float = DEFAULT_PIT_LOSS_S,
) -> tuple[bool, str]:
    """Determine if an undercut attempt is viable.

    Returns (is_viable, reasoning_string).

    Undercut is viable when:
    - Gap to car ahead < pit_loss − (fresh_tire_gain × FRESH_TIRE_ADVANTAGE_LAPS)
    - Driver's tires are more degraded than car ahead's (bigger gain from fresh rubber)

    Standard F1 undercut math:
        fresh_gain_total = FRESH_TIRE_GAIN_PER_LAP_S × FRESH_TIRE_ADVANTAGE_LAPS
        net_pit_loss     = pit_loss − fresh_gain_total
        undercut_viable  = gap_to_ahead < net_pit_loss
    """
    gap = driver.gap_to_ahead
    if gap is None:
        return False, "Gap to car ahead unknown — cannot evaluate undercut"

    fresh_gain_total = FRESH_TIRE_GAIN_PER_LAP_S * FRESH_TIRE_ADVANTAGE_LAPS
    net_pit_loss = pit_loss - fresh_gain_total

    viable = gap < net_pit_loss

    # Degradation edge: if driver is degrading faster, the gain is larger
    deg_advantage = ""
    if deg_driver and deg_ahead:
        delta_deg = deg_driver.deg_rate_per_lap - deg_ahead.deg_rate_per_lap
        if delta_deg > 0.05:
            viable = True  # tire condition seals the deal even if gap is marginal
            deg_advantage = (
                f" Driver degrading {delta_deg:.2f}s/lap faster than car ahead — "
                f"extra incentive to pit."
            )

    if viable:
        reasoning = (
            f"Undercut viable: gap to {car_ahead.name} is {gap:.2f}s, "
            f"net pit loss ~{net_pit_loss:.1f}s (pit loss {pit_loss:.0f}s − "
            f"fresh tire gain ~{fresh_gain_total:.1f}s over {FRESH_TIRE_ADVANTAGE_LAPS} laps)."
            + deg_advantage
        )
    else:
        reasoning = (
            f"Undercut not viable: gap to {car_ahead.name} is {gap:.2f}s, "
            f"but net pit loss is ~{net_pit_loss:.1f}s — would emerge behind."
        )

    return viable, reasoning


def evaluate_overcut(
    driver: DriverState,
    car_behind: DriverState,
    deg_driver: TireDegradation | None,
    deg_behind: TireDegradation | None,
    race_state: RaceState,
    pit_loss: float = DEFAULT_PIT_LOSS_S,
) -> tuple[bool, str]:
    """Determine if staying out (overcut) while car behind pits is viable.

    Overcut is viable when:
    - Driver's tires still have more pace to give than the pit-loss delta.
    - Car behind's fresh tires won't close the gap before driver pits next lap.

    This is a simplified model: overcut probability = gap_to_behind > pit_loss.
    """
    gap = car_behind.gap_to_ahead  # gap from car_behind to driver
    if gap is None:
        return False, "Gap to car behind unknown — cannot evaluate overcut"

    viable = gap > pit_loss

    # If car behind is degrading badly, they lose time before they can attack
    deg_boost = ""
    if deg_behind and deg_behind.deg_rate_per_lap > 0.2:
        viable = True
        deg_boost = (
            f" Car behind degrading at {deg_behind.deg_rate_per_lap:.2f}s/lap "
            f"— overcut gains time naturally."
        )

    if viable:
        reasoning = (
            f"Overcut viable: {car_behind.name} is {gap:.2f}s behind — "
            f"more than the ~{pit_loss:.0f}s pit loss. Stay out, extend stint, "
            f"and pit when {car_behind.name} is on worn tires."
            + deg_boost
        )
    else:
        reasoning = (
            f"Overcut risky: {car_behind.name} is only {gap:.2f}s behind "
            f"— they may emerge ahead after pitting first."
        )

    return viable, reasoning


def recommend_compound(
    laps_remaining: int,
    weather: WeatherState,
    available_compounds: list[TireCompound] | None = None,
) -> TireCompound:
    """Recommend the optimal tire compound for the remaining laps.

    Simplified expected life ranges (vary significantly by track):
      SOFT:         ~10–18 laps
      MEDIUM:       ~18–30 laps
      HARD:         ~28–40 laps
      INTERMEDIATE: wet/drying conditions
      WET:          heavy rain

    Args:
        laps_remaining:     Laps left after the pit stop.
        weather:            Current weather snapshot.
        available_compounds: Restrict to these compounds (None = all dry).
    """
    if weather.rainfall:
        return TireCompound.WET

    # Rain threat but not raining yet — could be intermediates soon
    # (caller provides weather history for full assessment via weather.py)

    allowed = set(available_compounds or [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD])

    # Simple rule-based selection
    if laps_remaining <= 15 and TireCompound.SOFT in allowed:
        return TireCompound.SOFT
    if laps_remaining <= 28 and TireCompound.MEDIUM in allowed:
        return TireCompound.MEDIUM
    if TireCompound.HARD in allowed:
        return TireCompound.HARD

    # Fall back to whatever is available in order of preference
    for c in (TireCompound.MEDIUM, TireCompound.SOFT, TireCompound.HARD):
        if c in allowed:
            return c

    return TireCompound.UNKNOWN


def build_strategy_recommendation(
    driver: DriverState,
    race_state: RaceState,
    deg: TireDegradation | None,
    weather_history: list[WeatherState],
    pit_loss: float = DEFAULT_PIT_LOSS_S,
) -> StrategyRecommendation:
    """Synthesise all analysis data into a StrategyRecommendation.

    This is the main function the Strategy Agent calls after gathering
    tire deg, weather, and historical context.
    """
    current_lap = race_state.current_lap
    total_laps = race_state.total_laps
    laps_remaining = total_laps - current_lap

    weather_events = detect_weather_changes(weather_history)
    rain_incoming = any(e.event == WeatherEvent.RAIN_THREAT for e in weather_events)
    rain_now = race_state.weather.rainfall

    # --- Determine action ---
    action: str
    reasoning_parts: list[str] = []
    confidence: float = 0.7

    # Rain override — always pit if rain starts
    if rain_now and driver.tire_compound not in (TireCompound.INTERMEDIATE, TireCompound.WET):
        action = "PIT_NOW"
        reasoning_parts.append("Rain detected — switch to intermediate/wet tires immediately.")
        confidence = 0.95
        recommended_compound = recommend_compound(laps_remaining, race_state.weather)

    # Cliff imminent — pit now or very soon
    elif deg and deg.laps_remaining_estimate is not None and deg.laps_remaining_estimate <= 2:
        action = "PIT_NOW"
        reasoning_parts.append(
            f"Tire cliff imminent in ~{deg.laps_remaining_estimate} lap(s) "
            f"(deg rate {deg.deg_rate_per_lap:.3f}s/lap on {deg.compound.value})."
        )
        confidence = 0.85
        recommended_compound = recommend_compound(laps_remaining, race_state.weather)

    # Still in viable window
    else:
        pit_window = calculate_optimal_pit_window(driver, deg, race_state, pit_loss)
        if pit_window and current_lap >= pit_window.earliest_lap:
            action = f"PIT_IN_{pit_window.ideal_lap - current_lap}_LAPS" if pit_window.ideal_lap > current_lap else "PIT_NOW"
            reasoning_parts.append(
                f"Optimal pit window: laps {pit_window.earliest_lap}–{pit_window.latest_lap} "
                f"(ideal lap {pit_window.ideal_lap})."
            )
            confidence = 0.72
        else:
            action = "STAY_OUT"
            reasoning_parts.append("No immediate pit trigger — current tires still viable.")
            confidence = 0.65
        recommended_compound = recommend_compound(laps_remaining, race_state.weather)

    # Rain threat flag — lower confidence, mention it
    if rain_incoming and not rain_now:
        reasoning_parts.append("Rain threat detected — monitor weather closely.")
        confidence = min(confidence, 0.6)

    # Evaluate undercut/overcut vs immediate neighbours
    undercut_viable = False
    overcut_viable = False
    sorted_drivers = sorted(race_state.drivers, key=lambda d: d.position)

    driver_idx = next(
        (i for i, d in enumerate(sorted_drivers) if d.driver_number == driver.driver_number), None
    )

    if driver_idx is not None:
        if driver_idx > 0:
            car_ahead = sorted_drivers[driver_idx - 1]
            undercut_viable, uc_reason = evaluate_undercut(driver, car_ahead, deg, None, pit_loss)
            if undercut_viable:
                reasoning_parts.append(uc_reason)

        if driver_idx < len(sorted_drivers) - 1:
            car_behind = sorted_drivers[driver_idx + 1]
            overcut_viable, oc_reason = evaluate_overcut(driver, car_behind, None, None, race_state, pit_loss)
            if overcut_viable:
                reasoning_parts.append(oc_reason)

    # Stint length note
    if driver.stint_length:
        reasoning_parts.append(
            f"Driver is on lap {driver.stint_length} of {driver.tire_compound.value} tires."
        )

    reasoning = " ".join(reasoning_parts) if reasoning_parts else "Insufficient data for analysis."

    # Determine pit window tuple for the model
    pw = calculate_optimal_pit_window(driver, deg, race_state, pit_loss)
    pit_window_tuple = (pw.earliest_lap, pw.latest_lap) if pw else None

    logger.info(
        "Strategy recommendation built",
        extra={
            "driver": driver.driver_number,
            "action": action,
            "confidence": confidence,
        },
    )

    return StrategyRecommendation(
        driver_number=driver.driver_number,
        recommended_action=action,
        recommended_compound=recommended_compound,
        optimal_pit_window=pit_window_tuple,
        undercut_viable=undercut_viable,
        overcut_viable=overcut_viable,
        reasoning=reasoning,
        confidence=round(confidence, 2),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _min_laps_for_compound(compound: TireCompound) -> int:
    """Minimum laps you'd want on a fresh tire to justify the stop."""
    match compound:
        case TireCompound.SOFT:
            return 8
        case TireCompound.MEDIUM:
            return 12
        case TireCompound.HARD:
            return 15
        case _:
            return 10

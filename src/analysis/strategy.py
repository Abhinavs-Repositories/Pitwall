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
        earliest = max(earliest_viable, cliff - 4)
        latest = min(latest_viable, cliff - 1)  # always pit before the cliff
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

    Undercut mechanics: you pit first, spend ~22s in the pit lane, then do
    2-3 laps on fresh rubber gaining ~0.8s/lap vs the car ahead on worn
    tires.  The undercut works when the gap to the car ahead is *less* than
    the total fresh-tire advantage you'll accumulate — i.e. the gap is small
    enough that the pace delta lets you jump them before they pit.

    Additionally, if the car ahead has higher degradation, their lap times
    worsen each lap they stay out, adding to the effective gain.
    """
    gap = driver.gap_to_ahead
    if gap is None:
        return False, "Gap to car ahead unknown — cannot evaluate undercut"

    fresh_gain_total = FRESH_TIRE_GAIN_PER_LAP_S * FRESH_TIRE_ADVANTAGE_LAPS

    # Add degradation delta: if car ahead is degrading, they lose extra time
    # while you're on fresh rubber
    deg_extra = 0.0
    deg_advantage = ""
    if deg_ahead and deg_ahead.deg_rate_per_lap > 0:
        deg_extra = deg_ahead.deg_rate_per_lap * FRESH_TIRE_ADVANTAGE_LAPS
    if deg_driver and deg_ahead:
        delta_deg = deg_driver.deg_rate_per_lap - deg_ahead.deg_rate_per_lap
        if delta_deg > 0.05:
            deg_advantage = (
                f" Driver degrading {delta_deg:.2f}s/lap faster than car ahead — "
                f"extra incentive to pit."
            )

    effective_gain = fresh_gain_total + deg_extra
    viable = gap < effective_gain

    if viable:
        reasoning = (
            f"Undercut viable: gap to {car_ahead.name} is {gap:.2f}s, "
            f"fresh tire advantage ~{effective_gain:.1f}s over "
            f"{FRESH_TIRE_ADVANTAGE_LAPS} laps "
            f"(pace gain {fresh_gain_total:.1f}s + deg advantage {deg_extra:.1f}s)."
            + deg_advantage
        )
    else:
        reasoning = (
            f"Undercut not viable: gap to {car_ahead.name} is {gap:.2f}s, "
            f"exceeds fresh tire advantage of ~{effective_gain:.1f}s."
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

    Overcut mechanics: the car behind pits first and loses ~22s in the pit
    lane.  You stay out for 1-2 extra laps on old tires, losing deg_rate
    per lap.  Then you pit yourself.  The overcut works when the car
    behind loses more time from their pit stop than you lose from staying
    out on degrading tires — i.e. you emerge ahead after you both pit.

    Viable when: gap_to_behind < pit_loss - our_time_loss_from_extra_laps
    (the gap is small enough that their pit loss puts them behind you, and
    your tire degradation over the extra laps doesn't erase that advantage).
    """
    gap = car_behind.gap_to_ahead  # gap from car_behind to driver
    if gap is None:
        return False, "Gap to car behind unknown — cannot evaluate overcut"

    extra_laps = 2
    our_time_loss = (deg_driver.deg_rate_per_lap * extra_laps) if deg_driver else 0.0
    net_advantage = pit_loss - our_time_loss

    viable = gap < net_advantage

    # If car behind is degrading badly, their out-lap on cold tires is even
    # slower, making the overcut stronger
    deg_boost = ""
    if deg_behind and deg_behind.deg_rate_per_lap > 0.2:
        deg_boost = (
            f" Car behind degrading at {deg_behind.deg_rate_per_lap:.2f}s/lap "
            f"— their out-lap will be slow on cold tires."
        )

    if viable:
        reasoning = (
            f"Overcut viable: {car_behind.name} is {gap:.2f}s behind. "
            f"If they pit, they lose ~{pit_loss:.0f}s; staying out "
            f"{extra_laps} extra laps costs ~{our_time_loss:.1f}s in deg. "
            f"Net advantage ~{net_advantage:.1f}s > gap."
            + deg_boost
        )
    else:
        reasoning = (
            f"Overcut risky: {car_behind.name} is {gap:.2f}s behind "
            f"— pit loss ~{pit_loss:.0f}s minus ~{our_time_loss:.1f}s deg "
            f"= ~{net_advantage:.1f}s net, not enough margin."
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
        # Light rain on a warm track → intermediates (most common rain scenario)
        # Heavy rain or cold track → full wets
        if weather.track_temp is not None and weather.track_temp > 25:
            return TireCompound.INTERMEDIATE
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

    # --- Data-quality multiplier for confidence ---
    # Scale confidence by how much clean data the deg regression is based on
    data_quality = 1.0
    if deg and deg.current_stint_laps:
        data_quality = min(1.0, deg.current_stint_laps / 15)

    # --- Compute pit window once (reused for action + return value) ---
    pit_window = calculate_optimal_pit_window(driver, deg, race_state, pit_loss)

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
        confidence = 0.85 * data_quality
        recommended_compound = recommend_compound(laps_remaining, race_state.weather)

    # Still in viable window
    else:
        if pit_window and current_lap >= pit_window.earliest_lap:
            action = f"PIT_IN_{pit_window.ideal_lap - current_lap}_LAPS" if pit_window.ideal_lap > current_lap else "PIT_NOW"
            reasoning_parts.append(
                f"Optimal pit window: laps {pit_window.earliest_lap}–{pit_window.latest_lap} "
                f"(ideal lap {pit_window.ideal_lap})."
            )
            confidence = 0.72 * data_quality
        else:
            action = "STAY_OUT"
            reasoning_parts.append("No immediate pit trigger — current tires still viable.")
            confidence = 0.65 * data_quality
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
            overcut_viable, oc_reason = evaluate_overcut(driver, car_behind, deg, None, race_state, pit_loss)
            if overcut_viable:
                reasoning_parts.append(oc_reason)

    # Stint length note
    if driver.stint_length:
        reasoning_parts.append(
            f"Driver is on lap {driver.stint_length} of {driver.tire_compound.value} tires."
        )

    reasoning = " ".join(reasoning_parts) if reasoning_parts else "Insufficient data for analysis."

    pit_window_tuple = (pit_window.earliest_lap, pit_window.latest_lap) if pit_window else None

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

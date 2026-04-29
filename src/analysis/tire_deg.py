"""Tire degradation analysis — pure Python / NumPy, zero LLM calls.

Key design decisions (from spec):
- Filter out pit in/out laps, safety car laps, and first 2 warm-up laps of each stint.
- Use linear regression on clean lap times to get degradation slope (seconds/lap).
- Tire cliff = when projected time exceeds stint_best + CLIFF_THRESHOLD_S.
- Compare degradation across drivers on the same compound for track-level insight.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from src.data.models import DriverState, LapData, RaceControlMessage, Stint, TireCompound, TireDegradation

logger = logging.getLogger(__name__)

# Seconds slower than stint-best before we call it a "cliff"
CLIFF_THRESHOLD_S: float = 2.5

# Minimum clean laps required for a reliable regression
MIN_CLEAN_LAPS: int = 6

# Warm-up laps to skip at the start of each stint
WARMUP_LAPS: int = 2

# Fuel burn effect: lighter car ≈ 0.06s/lap faster.  We add this back so
# the regression measures true tire degradation, not (tire_deg − fuel_gain).
FUEL_CORRECTION_PER_LAP: float = 0.06


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_degradation(
    lap_times: list[LapData],
    stint: Stint,
    safety_car_laps: set[int] | None = None,
) -> TireDegradation | None:
    """Compute degradation rate for a single stint.

    Returns None if there aren't enough clean laps for a reliable regression.

    Args:
        lap_times:        All lap data for a driver (may span multiple stints).
        stint:            The specific stint to analyse.
        safety_car_laps:  Set of lap numbers affected by safety car / VSC.
    """
    sc_laps = safety_car_laps or set()
    clean = _extract_clean_laps(lap_times, stint, sc_laps)

    if len(clean) < MIN_CLEAN_LAPS:
        logger.debug(
            "Not enough clean laps for degradation regression",
            extra={
                "stint_number": stint.stint_number,
                "compound": stint.compound.value,
                "clean_count": len(clean),
            },
        )
        return None

    lap_nums = np.array([l.lap_number for l in clean], dtype=float)
    times = np.array([l.lap_time for l in clean], dtype=float)

    # Normalise lap numbers relative to stint start so the intercept = lap-1 pace
    relative = lap_nums - stint.lap_start

    # Fuel correction: cars get ~0.06s/lap faster from fuel burn.  Add this
    # back so we measure true tire degradation, not (tire_deg − fuel_gain).
    corrected_times = times + FUEL_CORRECTION_PER_LAP * relative

    slope, intercept, r_squared = _linear_regression(relative, corrected_times)

    # If R² is very low, the regression is mostly noise — don't trust the cliff
    trust_regression = r_squared >= 0.3

    stint_best = float(corrected_times.min())
    current_stint_laps = int(lap_nums.max()) - stint.lap_start

    cliff_lap = _predict_cliff(slope, intercept, stint_best, stint.lap_start) if trust_regression else None

    laps_remaining: int | None = None
    if cliff_lap is not None:
        laps_remaining = max(0, cliff_lap - int(lap_nums.max()))

    logger.info(
        "Tire degradation computed",
        extra={
            "compound": stint.compound.value,
            "deg_rate": round(slope, 4),
            "r_squared": round(r_squared, 3),
            "cliff_lap": cliff_lap,
            "clean_laps_used": len(clean),
        },
    )

    return TireDegradation(
        driver_number=_driver_number_from_laps(lap_times),
        compound=stint.compound,
        deg_rate_per_lap=round(float(slope), 4),
        predicted_cliff_lap=cliff_lap,
        laps_remaining_estimate=laps_remaining,
        current_stint_laps=current_stint_laps,
    )


def predict_tire_cliff(
    lap_times: list[LapData],
    stint: Stint,
    safety_car_laps: set[int] | None = None,
) -> int | None:
    """Predict the lap number at which tire performance will fall off a cliff.

    Returns None if not enough data or if the cliff is not expected.
    """
    result = calculate_degradation(lap_times, stint, safety_car_laps)
    return result.predicted_cliff_lap if result else None


def compare_compound_performance(
    drivers: list[DriverState],
    compound: TireCompound,
    safety_car_laps: set[int] | None = None,
) -> list[_CompoundStat]:
    """Compare average pace and degradation across all drivers on a given compound.

    Returns a list of CompoundStat sorted by median pace (fastest first).
    """
    stats: list[_CompoundStat] = []
    for driver in drivers:
        for stint in driver.stints:
            if stint.compound != compound:
                continue
            sc_laps = safety_car_laps or set()
            clean = _extract_clean_laps(driver.lap_times, stint, sc_laps)
            if len(clean) < MIN_CLEAN_LAPS:
                continue
            times = np.array([l.lap_time for l in clean], dtype=float)
            lap_nums = np.array([l.lap_number for l in clean], dtype=float)
            relative = lap_nums - stint.lap_start
            corrected = times + FUEL_CORRECTION_PER_LAP * relative
            slope, _, _ = _linear_regression(relative, corrected)
            stats.append(
                _CompoundStat(
                    driver_number=driver.driver_number,
                    driver_name=driver.name,
                    compound=compound,
                    median_pace=float(np.median(times)),
                    deg_rate=round(float(slope), 4),
                    clean_laps=len(clean),
                )
            )

    stats.sort(key=lambda s: s.median_pace)
    return stats


def extract_safety_car_laps(
    race_control: list[RaceControlMessage],
    total_laps: int = 70,
) -> set[int]:
    """Derive a rough set of safety-car-affected lap numbers from RC messages.

    Strategy:
    1. Detect SC/VSC deployment and ending from message text + category.
    2. Try to extract a lap number from the message text (e.g. "LAP 23").
    3. Mark a conservative window: deployment lap through +3 laps (SC) or
       +2 laps (VSC), or until a SC-ending message is found.

    This is best-effort — OpenF1 RC messages don't carry a lap_number field.
    """
    sc_laps: set[int] = set()
    SC_WINDOW = 4   # conservative laps affected by full SC
    VSC_WINDOW = 3  # conservative laps affected by VSC

    for msg in race_control:
        cat = (msg.category or "").lower()
        text = msg.message.lower()

        is_sc = "safety car" in text or cat == "safetycar"
        is_vsc = "virtual safety car" in text or "vsc" in text
        is_ending = "ending" in text or "in this lap" in text or "withdrawn" in text

        if not (is_sc or is_vsc):
            continue
        if is_ending:
            continue  # SC ending messages — don't add more laps

        # Try to extract lap number from message text (e.g. "SAFETY CAR DEPLOYED - LAP 23")
        lap_match = re.search(r"lap\s+(\d{1,2})", text)
        if lap_match:
            deploy_lap = int(lap_match.group(1))
        else:
            # No lap number in message — skip rather than guess wildly
            continue

        window = VSC_WINDOW if is_vsc else SC_WINDOW
        for lap in range(deploy_lap, min(deploy_lap + window, total_laps + 1)):
            sc_laps.add(lap)

    return sc_laps


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_clean_laps(
    lap_times: list[LapData],
    stint: Stint,
    sc_laps: set[int],
) -> list[LapData]:
    """Return laps within this stint that are usable for regression."""
    clean: list[LapData] = []
    warmup_cutoff = stint.lap_start + WARMUP_LAPS

    for lap in lap_times:
        if lap.lap_number < warmup_cutoff:
            continue
        if lap.lap_number > stint.lap_end:
            continue
        if lap.is_pit_in_lap or lap.is_pit_out_lap:
            continue
        if lap.lap_number in sc_laps:
            continue
        if lap.lap_time is None or lap.lap_time <= 0:
            continue
        # Outlier guard: drop laps more than 15 s above stint minimum seen so far
        clean.append(lap)

    if not clean:
        return clean

    # Second pass: remove statistical outliers (> 5 s above current minimum)
    # 5s is generous for racing laps but filters SC/VSC/yellow-flag laps
    min_time = min(l.lap_time for l in clean)  # type: ignore[arg-type]
    clean = [l for l in clean if l.lap_time <= min_time + 5.0]  # type: ignore[operator]

    return clean


def _linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, r_squared) from ordinary least-squares regression."""
    if len(x) < 2:
        return 0.0, float(y.mean()) if len(y) else 0.0, 0.0
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    # Compute R² (coefficient of determination)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return slope, intercept, r_squared


def _predict_cliff(
    slope: float,
    intercept: float,
    stint_best: float,
    lap_start: int,
    max_look_ahead: int = 60,
) -> int | None:
    """Return the first relative lap where projected time exceeds best + CLIFF_THRESHOLD_S."""
    if slope <= 0:
        return None  # pace is improving or flat — no cliff

    for rel_lap in range(1, max_look_ahead + 1):
        projected = intercept + slope * rel_lap
        if projected >= stint_best + CLIFF_THRESHOLD_S:
            return lap_start + rel_lap

    return None


def _driver_number_from_laps(lap_times: list[LapData]) -> int:
    """Best-effort: laps don't carry driver_number in the model, return 0 as sentinel."""
    return 0


# ---------------------------------------------------------------------------
# Data class for compound comparison results
# ---------------------------------------------------------------------------


@dataclass
class _CompoundStat:
    driver_number: int
    driver_name: str
    compound: TireCompound
    median_pace: float
    deg_rate: float
    clean_laps: int


# Public alias
CompoundStat = _CompoundStat

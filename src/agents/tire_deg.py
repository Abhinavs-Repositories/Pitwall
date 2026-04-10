"""Tire Degradation Agent — analyses lap time trends and predicts tire cliffs.

Reads race_state from shared state, runs pure-Python analysis,
and writes tire_degradations keyed by driver_number (as string).
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.state import AgentState
from src.analysis.tire_deg import calculate_degradation, compare_compound_performance, extract_safety_car_laps
from src.data.models import TireDegradation

logger = logging.getLogger(__name__)


def tire_deg_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: compute degradation for all (or targeted) drivers."""
    race_state = state.race_state
    if not race_state:
        return {"errors": state.errors + ["tire_deg_node: no race_state available"]}

    sc_laps = extract_safety_car_laps(race_state.race_control)

    # Determine which drivers to analyse
    drivers_to_analyse = race_state.drivers
    if state.target_drivers:
        drivers_to_analyse = [
            d for d in race_state.drivers if d.driver_number in state.target_drivers
        ]
        if not drivers_to_analyse:
            # Fall back to all drivers if target_drivers don't match
            drivers_to_analyse = race_state.drivers

    degradations: dict[str, TireDegradation] = {}

    for driver in drivers_to_analyse:
        if not driver.stints:
            continue

        # Analyse the most recent (current) stint
        current_stint = max(driver.stints, key=lambda s: s.stint_number)
        deg = calculate_degradation(
            lap_times=driver.lap_times,
            stint=current_stint,
            safety_car_laps=sc_laps,
        )
        if deg:
            # Override driver_number — calculate_degradation uses 0 as sentinel
            deg = deg.model_copy(update={"driver_number": driver.driver_number})
            degradations[str(driver.driver_number)] = deg

    agents_used = list(state.agents_used) + ["tire_degradation"]

    logger.info(
        "Tire degradation computed for %d drivers", len(degradations),
        extra={"drivers": list(degradations.keys())},
    )

    return {"tire_degradations": degradations, "agents_used": agents_used}

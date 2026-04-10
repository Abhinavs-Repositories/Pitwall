"""Strategy Agent (The Brain) — synthesises all data into pit recommendations.

Combines race state + tire degradation + weather + historical context
using the pure-Python analysis functions (no LLM for the math).
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.state import AgentState
from src.analysis.strategy import build_strategy_recommendation
from src.data.models import StrategyRecommendation, TireDegradation

logger = logging.getLogger(__name__)


def strategy_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: build StrategyRecommendation for target drivers."""
    race_state = state.race_state
    if not race_state:
        return {"errors": state.errors + ["strategy_node: no race_state"]}

    # Which drivers to recommend for
    if state.target_drivers:
        target = [d for d in race_state.drivers if d.driver_number in state.target_drivers]
    else:
        # Default: all drivers, capped at top 10 to avoid huge state
        sorted_drivers = sorted(race_state.drivers, key=lambda d: d.position)
        target = sorted_drivers[:10]

    recommendations: dict[str, StrategyRecommendation] = {}

    # Track-level pit loss override from RAG
    pit_loss = float(state.track_characteristics.get("pit_loss_seconds", 22))

    for driver in target:
        deg: TireDegradation | None = state.tire_degradations.get(str(driver.driver_number))
        try:
            rec = build_strategy_recommendation(
                driver=driver,
                race_state=race_state,
                deg=deg,
                weather_history=state.weather_history,
                pit_loss=pit_loss,
            )
            recommendations[str(driver.driver_number)] = rec
        except Exception as exc:
            logger.warning(
                "Strategy recommendation failed for driver %s: %s",
                driver.driver_number,
                exc,
            )

    agents_used = list(state.agents_used) + ["strategy"]

    return {
        "strategy_recommendations": recommendations,
        "agents_used": agents_used,
    }

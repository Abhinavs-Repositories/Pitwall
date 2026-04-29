"""Race State Agent — fetches and structures live/historical race data from OpenF1.

Populates state.race_state with a fully hydrated RaceState object.
All data is cached in SQLite on first fetch (historical data never changes).
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.state import AgentState
from src.data.openf1_client import OpenF1Client
from src.data.race_builder import RaceBuilder

logger = logging.getLogger(__name__)


async def race_state_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: fetch race data and build a RaceState."""
    session_key = state.session_key
    current_lap = state.current_lap

    if not session_key:
        logger.warning("race_state_node called with no session_key")
        return {"errors": state.errors + ["No session_key provided"]}

    try:
        async with OpenF1Client() as client:
            builder = RaceBuilder(client)
            race_state = await builder.build(
                session_key=session_key,
                up_to_lap=current_lap or None,
            )

        return {"race_state": race_state, "agents_used": ["race_state"]}

    except Exception as exc:
        logger.error("race_state_node failed: %s", exc, exc_info=True)
        return {"errors": [f"Race state fetch error: {exc}"]}

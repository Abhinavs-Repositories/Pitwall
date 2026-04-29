"""Strategy RAG Agent — retrieves historical race strategies from Qdrant.

Uses the current race context (track, conditions) to find relevant
historical precedents that inform the Strategy Agent's recommendation.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.state import AgentState
from src.rag.retriever import StrategyRetriever

logger = logging.getLogger(__name__)


async def strategy_rag_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: retrieve historical strategies + track characteristics."""
    race_state = state.race_state
    user_message = state.user_message

    retriever = StrategyRetriever()
    await retriever.init()

    try:
        # Build a rich query from available context
        query_parts = [user_message] if user_message else ["F1 race strategy"]
        track_name = ""

        if race_state:
            track_name = race_state.track_name
            query_parts.append(f"at {track_name}")
            if race_state.weather.rainfall:
                query_parts.append("wet conditions rain")

        query = " ".join(query_parts)

        # Fetch historical strategies
        historical = await retriever.query_historical_strategies(
            query=query,
            track=track_name or None,
            top_k=3,
        )

        # Fetch track characteristics
        track_chars: dict[str, Any] = {}
        if track_name:
            data = retriever.get_track_characteristics(track_name)
            if data:
                track_chars = data

        return {
            "historical_context": historical,
            "track_characteristics": track_chars,
            "agents_used": ["strategy_rag"],
        }

    except Exception as exc:
        logger.error("strategy_rag_node failed: %s", exc, exc_info=True)
        return {
            "historical_context": [],
            "track_characteristics": {},
            "errors": [f"RAG error: {exc}"],
            "agents_used": ["strategy_rag"],
        }

    finally:
        await retriever.close()

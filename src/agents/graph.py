"""LangGraph state graph — wires all 6 agents into a single callable.

Graph topology (mirrors the spec):
                                         ┌─────────────┐
                                    ┌───>│  race_state  │
                                    │    └──────┬──────┘
                                    │           │
[router] ──> dispatch ─────────────>│    ┌──────▼──────┐
                                    │    │  tire_deg    │
                                    │    └──────┬──────┘
                                    │           │
                      ┌─────────────>│    ┌──────▼──────┐
                      │             └───>│   weather    │
                      │                  └──────┬──────┘
                      │                         │
                      │                  ┌──────▼──────────┐
                      │                  │  strategy_rag    │ (only for strategy/historical)
                      │                  └──────┬──────────┘
                      │                         │
                      │                  ┌──────▼──────┐
                      │                  │   strategy   │ (only for strategy/comparison)
                      │                  └──────┬──────┘
                      │                         │
                      └─────────────────────────>│
                                         ┌──────▼──────┐
                                         │  explainer   │
                                         └─────────────┘

Usage::

    from src.agents.graph import build_graph

    graph = build_graph()
    result = await graph.ainvoke({
        "session_key": 9158,
        "current_lap": 34,
        "user_message": "Should Verstappen pit now?",
    })
    print(result["final_response"])
"""

from __future__ import annotations

import logging
from typing import Any

# ---------------------------------------------------------------------------
# Patch logging.Logger.makeRecord before ANY agent logger is used.
# LangGraph runs nodes in thread executors; LangChain injects context that
# includes "message" as an extra key, which the default makeRecord blocks.
# ---------------------------------------------------------------------------
def _patch_logging() -> None:
    def _safe_make_record(self, name, level, fn, lno, msg, args, exc_info,
                          func=None, extra=None, sinfo=None):
        rv = logging.LogRecord(name, level, fn, lno, msg, args, exc_info, func, sinfo)
        if extra:
            reserved = {"message", "asctime"} | set(rv.__dict__.keys())
            for key, value in extra.items():
                if key not in reserved:
                    rv.__dict__[key] = value
        return rv
    logging.Logger.makeRecord = _safe_make_record

_patch_logging()
# ---------------------------------------------------------------------------

from langgraph.graph import END, START, StateGraph

from src.agents.explainer import explainer_node
from src.agents.race_state import race_state_node
from src.agents.router import router_node
from src.agents.state import AgentState
from src.agents.strategy import strategy_node
from src.agents.strategy_rag import strategy_rag_node
from src.agents.tire_deg import tire_deg_node
from src.agents.weather import weather_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional routing logic
# ---------------------------------------------------------------------------

def _after_router(state: AgentState) -> str:
    """Short-circuit off_topic queries directly to explainer."""
    if state.query_type == "off_topic":
        return "explainer"
    return "race_state"


def _fan_out_after_race_state(state: AgentState) -> list[str]:
    """Fan out to independent analysis nodes in parallel.

    tire_deg, weather, and strategy_rag don't depend on each other —
    they all read from race_state and can run concurrently.
    """
    targets = ["weather"]  # always run weather

    if state.query_type in ("tire_analysis", "strategy", "comparison"):
        targets.append("tire_deg")

    if state.query_type in ("strategy", "historical", "comparison"):
        targets.append("strategy_rag")

    return targets


def _needs_strategy(state: AgentState) -> str:
    """Route to strategy brain only for queries that produce recommendations."""
    if state.query_type in ("strategy", "comparison"):
        return "strategy"
    return "explainer"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph state graph."""
    builder = StateGraph(AgentState)

    # Add all agent nodes
    builder.add_node("router", router_node)
    builder.add_node("race_state", race_state_node)
    builder.add_node("tire_deg", tire_deg_node)
    builder.add_node("weather", weather_node)
    builder.add_node("strategy_rag", strategy_rag_node)
    builder.add_node("strategy", strategy_node)
    builder.add_node("explainer", explainer_node)

    # Entry point
    builder.add_edge(START, "router")

    # Router → off_topic short-circuit or Race State
    builder.add_conditional_edges(
        "router",
        _after_router,
        {"explainer": "explainer", "race_state": "race_state"},
    )

    # Race State → fan out to independent analysis nodes in parallel
    # tire_deg, weather, and strategy_rag are independent — run concurrently
    builder.add_conditional_edges(
        "race_state",
        _fan_out_after_race_state,
        ["tire_deg", "weather", "strategy_rag"],
    )

    # All analysis nodes fan in → strategy decision (or straight to explainer)
    builder.add_conditional_edges("tire_deg", _needs_strategy, {"strategy": "strategy", "explainer": "explainer"})
    builder.add_conditional_edges("weather", _needs_strategy, {"strategy": "strategy", "explainer": "explainer"})
    builder.add_conditional_edges("strategy_rag", _needs_strategy, {"strategy": "strategy", "explainer": "explainer"})

    # Strategy → Explainer (always)
    builder.add_edge("strategy", "explainer")

    # Explainer → END
    builder.add_edge("explainer", END)

    graph = builder.compile()
    logger.info("Pitwall-AI agent graph compiled successfully")
    return graph


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class PitwallAgent:
    """High-level wrapper around the compiled LangGraph."""

    def __init__(self) -> None:
        self._graph = build_graph()

    async def chat(
        self,
        session_key: int,
        current_lap: int,
        message: str,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Process a user message and return the agent response.

        Returns a dict with keys:
            response (str)
            strategy_data (StrategyRecommendation | None)
            agents_used (list[str])
        """
        import time
        t0 = time.monotonic()

        initial_state = {
            "session_key": session_key,
            "current_lap": current_lap,
            "user_message": message,
            "conversation_history": conversation_history or [],
        }

        try:
            final_state = await self._graph.ainvoke(initial_state)
        except Exception as exc:
            logger.error("Agent graph error: %s", exc, exc_info=True)
            return {
                "response": f"I encountered an error while processing your request: {exc}",
                "strategy_data": None,
                "agents_used": ["error"],
                "processing_time_ms": round((time.monotonic() - t0) * 1000, 1),
            }

        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

        # Extract primary strategy recommendation (first target driver or first available)
        strategy_data = None
        if final_state.get("strategy_recommendations"):
            recs = final_state["strategy_recommendations"]
            if final_state.get("target_drivers"):
                first_target = str(final_state["target_drivers"][0])
                strategy_data = recs.get(first_target) or next(iter(recs.values()), None)
            else:
                strategy_data = next(iter(recs.values()), None)

        return {
            "response": final_state.get("final_response", "No response generated"),
            "strategy_data": strategy_data,
            "agents_used": final_state.get("agents_used", []),
            "processing_time_ms": elapsed_ms,
        }

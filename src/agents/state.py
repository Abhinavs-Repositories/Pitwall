"""Shared LangGraph state schema for the Pitwall-AI agent graph.

All agents read from and write to this typed state dict.
LangGraph merges partial updates returned by each node.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
import operator

from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from src.data.models import (
    HistoricalStrategy,
    RaceState,
    StrategyRecommendation,
    TireDegradation,
    WeatherState,
)


class AgentState(BaseModel):
    """Full shared state passed between LangGraph nodes."""

    # ---- Input ----
    session_key: int = 0
    current_lap: int = 0
    user_message: str = ""
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)

    # ---- Router output ----
    query_type: str = ""  # race_status | tire_analysis | weather | strategy | comparison | historical
    target_drivers: list[int] = Field(default_factory=list)  # driver numbers mentioned in query

    # ---- Race State agent output ----
    race_state: Optional[RaceState] = None

    # ---- Tire Degradation agent output ----
    # Keyed by driver_number (as str for JSON-safe state)
    tire_degradations: dict[str, TireDegradation] = Field(default_factory=dict)

    # ---- Weather agent output ----
    weather_history: list[WeatherState] = Field(default_factory=list)
    weather_alert: str = ""  # human-readable alert if rain threat detected

    # ---- Strategy RAG agent output ----
    historical_context: list[HistoricalStrategy] = Field(default_factory=list)
    track_characteristics: dict[str, Any] = Field(default_factory=dict)

    # ---- Strategy agent output ----
    strategy_recommendations: dict[str, StrategyRecommendation] = Field(default_factory=dict)  # keyed by driver_number

    # ---- Explainer agent output (final) ----
    final_response: str = ""
    agents_used: Annotated[list[str], operator.add] = Field(default_factory=list)

    # ---- Error handling ----
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

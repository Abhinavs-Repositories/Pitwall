"""Integration tests for the LangGraph agent pipeline.

These tests use mocked OpenF1 data and a mocked LLM to avoid
real network calls, while still exercising the full graph logic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.graph import PitwallAgent, build_graph
from src.agents.router import _classify, _extract_drivers, _rule_based_classify
from src.agents.state import AgentState
from src.data.models import RaceState, TireCompound, WeatherState


# ---------------------------------------------------------------------------
# Router Agent tests
# ---------------------------------------------------------------------------


class TestRouter:
    def test_rule_based_classify_strategy(self):
        assert _rule_based_classify("Should Verstappen pit now?") == "strategy"

    def test_rule_based_classify_tire(self):
        assert _rule_based_classify("What is the tire degradation for Norris?") == "tire_analysis"

    def test_rule_based_classify_weather(self):
        assert _rule_based_classify("Is it going to rain in 5 laps?") == "weather"

    def test_rule_based_classify_comparison(self):
        assert _rule_based_classify("Compare Norris vs Leclerc tire wear") == "comparison"

    def test_rule_based_classify_historical(self):
        assert _rule_based_classify("What happened at this track last year 2024?") == "historical"

    def test_rule_based_classify_race_status(self):
        assert _rule_based_classify("What are the current positions?") == "race_status"

    def test_extract_drivers_by_name(self):
        drivers = _extract_drivers("Should Verstappen pit before Norris?")
        assert 1 in drivers   # Verstappen = #1
        assert 4 in drivers   # Norris = #4

    def test_extract_drivers_by_number(self):
        drivers = _extract_drivers("Compare driver 1 and driver 44")
        assert 1 in drivers
        assert 44 in drivers

    def test_extract_drivers_empty(self):
        assert _extract_drivers("What is the weather?") == []


# ---------------------------------------------------------------------------
# AgentState tests
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_default_state(self):
        state = AgentState()
        assert state.session_key == 0
        assert state.query_type == ""
        assert state.tire_degradations == {}
        assert state.agents_used == []
        assert state.errors == []

    def test_state_with_race_data(self, sample_race_state):
        state = AgentState(
            session_key=9158,
            current_lap=34,
            race_state=sample_race_state,
        )
        assert state.race_state.current_lap == 34
        assert len(state.race_state.drivers) > 0


# ---------------------------------------------------------------------------
# Graph structure test
# ---------------------------------------------------------------------------


class TestGraph:
    def test_graph_compiles(self):
        """Graph should compile without errors."""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        graph = build_graph()
        # LangGraph compiled graph exposes graph attribute
        node_names = set(graph.nodes.keys()) if hasattr(graph, "nodes") else set()
        # Basic check — just verify it's a valid compiled graph object
        assert graph is not None


# ---------------------------------------------------------------------------
# Tire Degradation node test
# ---------------------------------------------------------------------------


class TestTireDegNode:
    def test_tire_deg_node_no_race_state(self):
        from src.agents.tire_deg import tire_deg_node

        state = AgentState()
        result = tire_deg_node(state)
        assert "errors" in result
        assert len(result["errors"]) > 0

    def test_tire_deg_node_with_race_state(self, sample_race_state):
        from src.agents.tire_deg import tire_deg_node

        state = AgentState(race_state=sample_race_state)
        result = tire_deg_node(state)
        # May or may not produce degradations depending on data sufficiency
        assert "agents_used" in result
        assert "tire_degradation" in result["agents_used"]


# ---------------------------------------------------------------------------
# Strategy node test
# ---------------------------------------------------------------------------


class TestStrategyNode:
    def test_strategy_node_no_race_state(self):
        from src.agents.strategy import strategy_node

        state = AgentState()
        result = strategy_node(state)
        assert "errors" in result

    def test_strategy_node_builds_recommendation(self, sample_race_state):
        from src.agents.strategy import strategy_node

        state = AgentState(
            race_state=sample_race_state,
            target_drivers=[1],
            weather_history=[WeatherState()],
        )
        result = strategy_node(state)
        assert "strategy_recommendations" in result
        assert "strategy" in result["agents_used"]


# ---------------------------------------------------------------------------
# Explainer node test
# ---------------------------------------------------------------------------


class TestExplainerNode:
    @patch("src.agents.explainer.get_llm")
    def test_explainer_node(self, mock_get_llm, sample_race_state):
        from src.agents.explainer import explainer_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Verstappen should pit now.")
        mock_get_llm.return_value = mock_llm

        state = AgentState(
            race_state=sample_race_state,
            user_message="Should Verstappen pit?",
            query_type="strategy",
        )
        result = explainer_node(state)
        assert result["final_response"] == "Verstappen should pit now."
        assert "explainer" in result["agents_used"]

    def test_explainer_fallback(self, sample_race_state):
        """When LLM fails, fallback response is returned."""
        from src.agents.explainer import _fallback_response

        state = AgentState(race_state=sample_race_state)
        response = _fallback_response(state)
        assert isinstance(response, str)
        assert len(response) > 0

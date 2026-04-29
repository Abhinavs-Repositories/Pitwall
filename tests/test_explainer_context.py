"""End-to-end tests for the explainer context building.

Validates that pit stop info, stint history, tire data, and all prompt types
produce correct context for the LLM — preventing hallucination of pit data.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.explainer import _build_context, _build_prompt, explainer_node
from src.agents.router import _rule_based_classify
from src.agents.state import AgentState
from src.data.models import (
    DriverState,
    LapData,
    PitStop,
    RaceState,
    Stint,
    StrategyRecommendation,
    TireCompound,
    TireDegradation,
    WeatherState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_driver(
    number: int,
    name: str,
    team: str,
    position: int,
    compound: TireCompound,
    stint_length: int,
    gap: float | None,
    stints: list[Stint] | None = None,
    pit_stops: list[PitStop] | None = None,
    lap_times: list[LapData] | None = None,
) -> DriverState:
    return DriverState(
        driver_number=number,
        name=name,
        team=team,
        position=position,
        gap_to_leader=gap,
        last_lap_time=95.0,
        tire_compound=compound,
        stint_length=stint_length,
        stints=stints or [],
        pit_stops=pit_stops or [],
        lap_times=lap_times or [],
    )


@pytest.fixture()
def race_state_with_pits() -> RaceState:
    """Race at lap 34: VER pitted once, NOR pitted once, LEC pitted twice."""
    drivers = [
        _make_driver(
            1, "Max Verstappen", "Red Bull Racing", 1, TireCompound.HARD, 7, 0.0,
            stints=[
                Stint(stint_number=1, compound=TireCompound.MEDIUM, lap_start=1, lap_end=27),
                Stint(stint_number=2, compound=TireCompound.HARD, lap_start=28, lap_end=34),
            ],
            pit_stops=[
                PitStop(lap_number=27, stop_duration=22.3,
                        compound_before=TireCompound.MEDIUM, compound_after=TireCompound.HARD),
            ],
        ),
        _make_driver(
            4, "Lando Norris", "McLaren", 2, TireCompound.HARD, 16, 4.231,
            stints=[
                Stint(stint_number=1, compound=TireCompound.SOFT, lap_start=1, lap_end=18),
                Stint(stint_number=2, compound=TireCompound.HARD, lap_start=19, lap_end=34),
            ],
            pit_stops=[
                PitStop(lap_number=18, stop_duration=21.8,
                        compound_before=TireCompound.SOFT, compound_after=TireCompound.HARD),
            ],
        ),
        _make_driver(
            16, "Charles Leclerc", "Ferrari", 3, TireCompound.HARD, 5, 8.015,
            stints=[
                Stint(stint_number=1, compound=TireCompound.MEDIUM, lap_start=1, lap_end=15),
                Stint(stint_number=2, compound=TireCompound.HARD, lap_start=16, lap_end=29),
                Stint(stint_number=3, compound=TireCompound.HARD, lap_start=30, lap_end=34),
            ],
            pit_stops=[
                PitStop(lap_number=15, stop_duration=23.1,
                        compound_before=TireCompound.MEDIUM, compound_after=TireCompound.HARD),
                PitStop(lap_number=29, stop_duration=22.7,
                        compound_before=TireCompound.HARD, compound_after=TireCompound.HARD),
            ],
        ),
    ]
    return RaceState(
        session_key=9158,
        meeting_name="Bahrain Grand Prix",
        track_name="Bahrain International Circuit",
        current_lap=34,
        total_laps=57,
        drivers=drivers,
        weather=WeatherState(air_temp=28.4, track_temp=42.1, humidity=44.5, rainfall=False),
    )


@pytest.fixture()
def race_state_no_pits() -> RaceState:
    """Early race (lap 5): nobody has pitted yet."""
    drivers = [
        _make_driver(
            1, "Max Verstappen", "Red Bull Racing", 1, TireCompound.SOFT, 5, 0.0,
            stints=[Stint(stint_number=1, compound=TireCompound.SOFT, lap_start=1, lap_end=5)],
        ),
        _make_driver(
            11, "Sergio Perez", "Red Bull Racing", 2, TireCompound.MEDIUM, 5, 1.2,
            stints=[Stint(stint_number=1, compound=TireCompound.MEDIUM, lap_start=1, lap_end=5)],
        ),
        _make_driver(
            16, "Charles Leclerc", "Ferrari", 3, TireCompound.SOFT, 5, 2.8,
            stints=[Stint(stint_number=1, compound=TireCompound.SOFT, lap_start=1, lap_end=5)],
        ),
    ]
    return RaceState(
        session_key=9158,
        meeting_name="Bahrain Grand Prix",
        track_name="Bahrain International Circuit",
        current_lap=5,
        total_laps=57,
        drivers=drivers,
        weather=WeatherState(air_temp=28.4, track_temp=42.1, humidity=44.5, rainfall=False),
    )


@pytest.fixture()
def race_state_mixed() -> RaceState:
    """Mid-race: some drivers pitted, some haven't."""
    drivers = [
        _make_driver(
            1, "Max Verstappen", "Red Bull Racing", 1, TireCompound.HARD, 3, 0.0,
            stints=[
                Stint(stint_number=1, compound=TireCompound.MEDIUM, lap_start=1, lap_end=12),
                Stint(stint_number=2, compound=TireCompound.HARD, lap_start=13, lap_end=15),
            ],
            pit_stops=[
                PitStop(lap_number=12, stop_duration=22.0,
                        compound_before=TireCompound.MEDIUM, compound_after=TireCompound.HARD),
            ],
        ),
        _make_driver(
            11, "Sergio Perez", "Red Bull Racing", 2, TireCompound.SOFT, 15, 3.5,
            stints=[Stint(stint_number=1, compound=TireCompound.SOFT, lap_start=1, lap_end=15)],
        ),
        _make_driver(
            16, "Charles Leclerc", "Ferrari", 3, TireCompound.HARD, 5, 5.0,
            stints=[
                Stint(stint_number=1, compound=TireCompound.SOFT, lap_start=1, lap_end=10),
                Stint(stint_number=2, compound=TireCompound.HARD, lap_start=11, lap_end=15),
            ],
            pit_stops=[
                PitStop(lap_number=10, stop_duration=21.5,
                        compound_before=TireCompound.SOFT, compound_after=TireCompound.HARD),
            ],
        ),
    ]
    return RaceState(
        session_key=9158,
        meeting_name="Bahrain Grand Prix",
        track_name="Bahrain International Circuit",
        current_lap=15,
        total_laps=57,
        drivers=drivers,
        weather=WeatherState(air_temp=30.0, track_temp=45.0, humidity=40.0, rainfall=False),
    )


# ---------------------------------------------------------------------------
# 1. Pit stop context accuracy
# ---------------------------------------------------------------------------


class TestPitStopContext:
    """Verify pit stop and stint data appears correctly in LLM context."""

    def test_pitted_driver_shows_stop_info(self, race_state_with_pits):
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="Who has pitted?",
            query_type="race_status",
        )
        context = _build_context(state)

        # Verstappen's pit stop
        assert "Stops: Lap 27 (MEDIUM\u2192HARD)" in context
        # Norris's pit stop
        assert "Stops: Lap 18 (SOFT\u2192HARD)" in context
        # Leclerc's two pit stops
        assert "Lap 15 (MEDIUM\u2192HARD)" in context
        assert "Lap 29 (HARD\u2192HARD)" in context

    def test_no_pit_drivers_marked_correctly(self, race_state_no_pits):
        state = AgentState(
            race_state=race_state_no_pits,
            user_message="Has anyone pitted yet?",
            query_type="race_status",
        )
        context = _build_context(state)

        assert context.count("No pit stops yet") == 3

    def test_mixed_pit_status(self, race_state_mixed):
        state = AgentState(
            race_state=race_state_mixed,
            user_message="Who has pitted already?",
            query_type="race_status",
        )
        context = _build_context(state)

        # Verstappen pitted
        assert "Stops: Lap 12 (MEDIUM\u2192HARD)" in context
        # Perez has NOT pitted
        assert "Perez" in context
        # Leclerc pitted
        assert "Stops: Lap 10 (SOFT\u2192HARD)" in context

        # Verify Perez line specifically says no pit stops
        lines = context.split("\n")
        perez_line = [l for l in lines if "Perez" in l][0]
        assert "No pit stops yet" in perez_line

    def test_stint_history_shown_for_multi_stint(self, race_state_with_pits):
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="Show me stint history",
            query_type="race_status",
        )
        context = _build_context(state)

        # Verstappen: 2 stints
        assert "S1:MEDIUM L1-27" in context
        assert "S2:HARD L28-34" in context

        # Leclerc: 3 stints
        assert "S1:MEDIUM L1-15" in context
        assert "S2:HARD L16-29" in context
        assert "S3:HARD L30-34" in context

    def test_stint_history_hidden_for_single_stint(self, race_state_no_pits):
        """Single-stint drivers should NOT show stint history (redundant)."""
        state = AgentState(
            race_state=race_state_no_pits,
            user_message="What tires is everyone on?",
            query_type="race_status",
        )
        context = _build_context(state)
        assert "Stints:" not in context

    def test_stint_lap_label_not_ambiguous(self, race_state_with_pits):
        """The label should say 'stint lap N' not just 'Lap N' to avoid confusion."""
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="Current tire status?",
            query_type="race_status",
        )
        context = _build_context(state)
        assert "stint lap" in context


# ---------------------------------------------------------------------------
# 2. Race status prompts
# ---------------------------------------------------------------------------


class TestRaceStatusPrompts:
    """Queries about positions, gaps, standings."""

    def test_top_5_standings(self, race_state_with_pits):
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="What are the current top 5 standings?",
            query_type="race_status",
        )
        context = _build_context(state)

        assert "P1" in context
        assert "P2" in context
        assert "P3" in context
        assert "Verstappen" in context
        assert "Norris" in context
        assert "Leclerc" in context

    def test_gaps_shown(self, race_state_with_pits):
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="What's the gap to the leader?",
            query_type="race_status",
        )
        context = _build_context(state)

        assert "LEADER" in context
        assert "+4.231s" in context
        assert "+8.015s" in context

    def test_lap_count_shown(self, race_state_with_pits):
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="What lap are we on?",
            query_type="race_status",
        )
        context = _build_context(state)
        assert "Lap 34/57" in context

    def test_race_name_shown(self, race_state_with_pits):
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="Which race is this?",
            query_type="race_status",
        )
        context = _build_context(state)
        assert "Bahrain Grand Prix" in context


# ---------------------------------------------------------------------------
# 3. Tire analysis prompts
# ---------------------------------------------------------------------------


class TestTireAnalysisPrompts:
    def test_tire_compound_in_context(self, race_state_with_pits):
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="What tires is Verstappen on?",
            query_type="tire_analysis",
        )
        context = _build_context(state)
        assert "HARD" in context
        assert "stint lap 7" in context

    def test_degradation_data_in_context(self, race_state_with_pits):
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="What's the tire deg for Verstappen?",
            query_type="tire_analysis",
            tire_degradations={
                "1": TireDegradation(
                    driver_number=1,
                    compound=TireCompound.HARD,
                    deg_rate_per_lap=0.045,
                    laps_remaining_estimate=12,
                    current_stint_laps=7,
                ),
            },
        )
        context = _build_context(state)
        assert "Deg: 0.045s/lap" in context
        assert "~12 laps to cliff" in context

    def test_pit_info_included_in_tire_analysis(self, race_state_with_pits):
        """Tire analysis should also show pit stop data."""
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="Tire analysis for all drivers",
            query_type="tire_analysis",
        )
        context = _build_context(state)
        assert "Stops:" in context or "No pit stops yet" in context


# ---------------------------------------------------------------------------
# 4. Weather prompts
# ---------------------------------------------------------------------------


class TestWeatherPrompts:
    def test_weather_context(self, race_state_with_pits):
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="What's the weather like?",
            query_type="weather",
        )
        context = _build_context(state)
        assert "Air 28.4" in context
        assert "Track 42.1" in context
        assert "Rain: No" in context

    def test_weather_alert_shown(self, race_state_with_pits):
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="Is it going to rain?",
            query_type="weather",
            weather_alert="Rain probability 60% in 5 laps",
        )
        context = _build_context(state)
        assert "Rain probability 60%" in context

    def test_weather_does_not_include_standings(self, race_state_with_pits):
        """Weather queries should NOT include standings (per _CONTEXT_SECTIONS)."""
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="What is the weather?",
            query_type="weather",
        )
        context = _build_context(state)
        assert "standings" not in context.lower()
        assert "P1" not in context


# ---------------------------------------------------------------------------
# 5. Strategy prompts
# ---------------------------------------------------------------------------


class TestStrategyPrompts:
    def test_strategy_includes_all_sections(self, race_state_with_pits):
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="Should Verstappen pit now?",
            query_type="strategy",
            strategy_recommendations={
                "1": StrategyRecommendation(
                    driver_number=1,
                    recommended_action="STAY_OUT",
                    reasoning="Hard tires still have life, no undercut threat.",
                    confidence=0.8,
                ),
            },
        )
        context = _build_context(state)
        assert "STAY_OUT" in context
        assert "80%" in context
        assert "Verstappen" in context
        assert "Stops:" in context  # pit info should be present

    def test_strategy_includes_pit_data(self, race_state_mixed):
        """Strategy recommendations need pit stop context."""
        state = AgentState(
            race_state=race_state_mixed,
            user_message="When should Perez pit?",
            query_type="strategy",
        )
        context = _build_context(state)
        perez_lines = [l for l in context.split("\n") if "Perez" in l]
        assert any("No pit stops yet" in l for l in perez_lines)


# ---------------------------------------------------------------------------
# 6. Comparison prompts
# ---------------------------------------------------------------------------


class TestComparisonPrompts:
    def test_comparison_includes_standings(self, race_state_with_pits):
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="Compare Verstappen vs Norris",
            query_type="comparison",
        )
        context = _build_context(state)
        assert "Verstappen" in context
        assert "Norris" in context
        assert "P1" in context
        assert "P2" in context


# ---------------------------------------------------------------------------
# 7. Historical prompts
# ---------------------------------------------------------------------------


class TestHistoricalPrompts:
    def test_historical_no_standings(self, race_state_with_pits):
        """Historical queries don't need current standings."""
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="What happened at this track last year?",
            query_type="historical",
        )
        context = _build_context(state)
        # Should have race summary but no standings
        assert "Bahrain Grand Prix" in context
        assert "P1" not in context

    def test_historical_with_context(self, race_state_with_pits):
        from src.data.models import HistoricalStrategy

        state = AgentState(
            race_state=race_state_with_pits,
            user_message="How did the winner approach strategy last year?",
            query_type="historical",
            historical_context=[
                HistoricalStrategy(
                    race_name="Bahrain Grand Prix",
                    year=2023,
                    track="Bahrain",
                    winner="Max Verstappen",
                    winner_strategy="M-H",
                    total_laps=57,
                    pit_stops_winner=1,
                    weather_conditions="Dry",
                    key_events="No safety car",
                    summary="Verstappen led from start to finish with a 1-stop strategy.",
                ),
            ],
        )
        context = _build_context(state)
        assert "2023" in context
        assert "M-H" in context


# ---------------------------------------------------------------------------
# 8. Off-topic prompts
# ---------------------------------------------------------------------------


class TestOffTopicPrompts:
    def test_off_topic_short_circuits(self):
        state = AgentState(
            user_message="Tell me a joke",
            query_type="off_topic",
        )
        result = explainer_node(state)
        assert "focused on F1" in result["final_response"]

    def test_off_topic_no_context_needed(self):
        state = AgentState(
            user_message="What's the capital of France?",
            query_type="off_topic",
        )
        context = _build_context(state)
        assert context == ""


# ---------------------------------------------------------------------------
# 9. Router classification for pit-related queries
# ---------------------------------------------------------------------------


class TestRouterClassification:
    """Verify the router classifies pit-related queries correctly."""

    @pytest.mark.parametrize("query,expected", [
        ("What are the current positions?", "race_status"),  # "position" is in F1 keywords
        ("Who has pitted already?", "strategy"),  # "pit" triggers strategy
        ("What tires is Verstappen on?", "tire_analysis"),
        ("Is it going to rain?", "weather"),
        ("Should Hamilton pit now?", "strategy"),
        ("Compare Norris vs Leclerc", "comparison"),
        ("What happened at this track in 2024?", "historical"),
        ("Tell me a joke", "off_topic"),
        ("What compound should Verstappen switch to?", "strategy"),
        ("How many stops has Leclerc made?", "strategy"),  # "stop" triggers strategy
    ])
    def test_classification(self, query, expected):
        result = _rule_based_classify(query)
        assert result == expected, f"Query '{query}' classified as '{result}', expected '{expected}'"


# ---------------------------------------------------------------------------
# 10. Full explainer node E2E with mocked LLM
# ---------------------------------------------------------------------------


class TestExplainerNodeE2E:
    """End-to-end tests through explainer_node with mocked LLM."""

    @patch("src.agents.explainer.get_llm")
    def test_pit_query_sends_pit_data_to_llm(self, mock_get_llm, race_state_with_pits):
        """The prompt sent to the LLM should contain pit stop data."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Verstappen pitted on lap 27.")
        mock_get_llm.return_value = mock_llm

        state = AgentState(
            race_state=race_state_with_pits,
            user_message="Who has pitted and on which lap?",
            query_type="race_status",
        )
        result = explainer_node(state)

        # Verify the LLM was called
        assert mock_llm.invoke.called
        # Check the prompt that was sent to the LLM
        call_args = mock_llm.invoke.call_args[0][0]  # messages list
        user_prompt = call_args[-1]["content"]  # last message is the user prompt
        assert "Stops: Lap 27" in user_prompt
        assert "Stops: Lap 18" in user_prompt
        assert "Lap 15 (MEDIUM\u2192HARD)" in user_prompt

    @patch("src.agents.explainer.get_llm")
    def test_no_pit_query_sends_no_pit_marker(self, mock_get_llm, race_state_no_pits):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Nobody has pitted yet.")
        mock_get_llm.return_value = mock_llm

        state = AgentState(
            race_state=race_state_no_pits,
            user_message="Has anyone pitted?",
            query_type="race_status",
        )
        result = explainer_node(state)

        call_args = mock_llm.invoke.call_args[0][0]
        user_prompt = call_args[-1]["content"]
        assert "No pit stops yet" in user_prompt

    @patch("src.agents.explainer.get_llm")
    def test_mixed_pit_query(self, mock_get_llm, race_state_mixed):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Verstappen and Leclerc have pitted. Perez has not."
        )
        mock_get_llm.return_value = mock_llm

        state = AgentState(
            race_state=race_state_mixed,
            user_message="Who has pitted and who hasn't?",
            query_type="race_status",
        )
        result = explainer_node(state)

        call_args = mock_llm.invoke.call_args[0][0]
        user_prompt = call_args[-1]["content"]

        # Verstappen pitted
        assert "Stops: Lap 12 (MEDIUM\u2192HARD)" in user_prompt
        # Perez has not
        lines = user_prompt.split("\n")
        perez_line = [l for l in lines if "Perez" in l][0]
        assert "No pit stops yet" in perez_line
        # Leclerc pitted
        assert "Stops: Lap 10 (SOFT\u2192HARD)" in user_prompt

    @patch("src.agents.explainer.get_llm")
    def test_strategy_query_has_full_context(self, mock_get_llm, race_state_with_pits):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Stay out on hards.")
        mock_get_llm.return_value = mock_llm

        state = AgentState(
            race_state=race_state_with_pits,
            user_message="Should Verstappen pit?",
            query_type="strategy",
            strategy_recommendations={
                "1": StrategyRecommendation(
                    driver_number=1,
                    recommended_action="STAY_OUT",
                    reasoning="Hard tires have 12 laps remaining.",
                    confidence=0.85,
                ),
            },
        )
        result = explainer_node(state)

        call_args = mock_llm.invoke.call_args[0][0]
        user_prompt = call_args[-1]["content"]
        assert "STAY_OUT" in user_prompt
        assert "85%" in user_prompt
        assert "Stops: Lap 27" in user_prompt

    @patch("src.agents.explainer.get_llm")
    def test_llm_failure_uses_fallback(self, mock_get_llm, race_state_with_pits):
        mock_get_llm.side_effect = Exception("LLM unavailable")

        state = AgentState(
            race_state=race_state_with_pits,
            user_message="Who has pitted?",
            query_type="race_status",
        )
        result = explainer_node(state)
        assert "Lap 34/57" in result["final_response"]

    def test_no_race_state_returns_error(self):
        state = AgentState(
            user_message="Who has pitted?",
            query_type="race_status",
            errors=["No session_key provided"],
        )
        result = explainer_node(state)
        assert "issues fetching" in result["final_response"]


# ---------------------------------------------------------------------------
# 11. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_driver_list(self):
        rs = RaceState(
            session_key=9158,
            meeting_name="Test GP",
            track_name="Test Circuit",
            current_lap=1,
            total_laps=50,
            drivers=[],
            weather=WeatherState(),
        )
        state = AgentState(race_state=rs, user_message="Status?", query_type="race_status")
        context = _build_context(state)
        assert "Current standings:" in context

    def test_driver_with_unknown_compound(self):
        driver = _make_driver(
            99, "Test Driver", "Test Team", 20, TireCompound.UNKNOWN, 0, 99.0,
        )
        rs = RaceState(
            session_key=1, meeting_name="Test", track_name="Test",
            current_lap=1, total_laps=50, drivers=[driver],
        )
        state = AgentState(race_state=rs, user_message="Status?", query_type="race_status")
        context = _build_context(state)
        assert "UNKNOWN" in context
        assert "No pit stops yet" in context

    def test_pit_stop_with_none_compounds(self):
        """Handles pit stops where compound_before/after is None."""
        driver = _make_driver(
            1, "Max Verstappen", "Red Bull", 1, TireCompound.HARD, 5, 0.0,
            stints=[
                Stint(stint_number=1, compound=TireCompound.SOFT, lap_start=1, lap_end=10),
                Stint(stint_number=2, compound=TireCompound.HARD, lap_start=11, lap_end=15),
            ],
            pit_stops=[
                PitStop(lap_number=10, stop_duration=22.0,
                        compound_before=None, compound_after=None),
            ],
        )
        rs = RaceState(
            session_key=1, meeting_name="Test", track_name="Test",
            current_lap=15, total_laps=50, drivers=[driver],
        )
        state = AgentState(race_state=rs, user_message="Status?", query_type="race_status")
        context = _build_context(state)
        assert "Lap 10 (?\u2192?)" in context

    def test_build_prompt_format(self, race_state_with_pits):
        state = AgentState(
            race_state=race_state_with_pits,
            user_message="Who has pitted?",
            query_type="race_status",
        )
        context = _build_context(state)
        prompt = _build_prompt("Who has pitted?", context, "race_status")
        assert "[Context]" in prompt
        assert "[Query type: race_status]" in prompt
        assert "[User question]" in prompt
        assert "Who has pitted?" in prompt

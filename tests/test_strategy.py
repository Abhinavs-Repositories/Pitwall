"""Tests for src/analysis/strategy.py and src/analysis/weather.py"""

import pytest

from src.analysis.strategy import (
    DEFAULT_PIT_LOSS_S,
    FRESH_TIRE_ADVANTAGE_LAPS,
    FRESH_TIRE_GAIN_PER_LAP_S,
    PitWindow,
    build_strategy_recommendation,
    calculate_optimal_pit_window,
    evaluate_overcut,
    evaluate_undercut,
    recommend_compound,
)
from src.analysis.weather import (
    WeatherEvent,
    detect_weather_changes,
    get_current_conditions_summary,
    is_rain_threat,
    recommend_tire_for_conditions,
)
from src.data.models import (
    DriverState,
    RaceState,
    TireCompound,
    TireDegradation,
    WeatherState,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_weather(
    air_temp: float = 28.0,
    track_temp: float = 42.0,
    humidity: float = 45.0,
    rainfall: bool = False,
    wind_speed: float = 3.0,
    wind_direction: int = 180,
) -> WeatherState:
    return WeatherState(
        air_temp=air_temp,
        track_temp=track_temp,
        humidity=humidity,
        rainfall=rainfall,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
    )


def _make_driver(
    number: int = 1,
    position: int = 1,
    compound: TireCompound = TireCompound.MEDIUM,
    stint_length: int = 18,
    gap_to_leader: float | None = 0.0,
    gap_to_ahead: float | None = None,
) -> DriverState:
    return DriverState(
        driver_number=number,
        name=f"Driver {number}",
        team="Team",
        position=position,
        tire_compound=compound,
        stint_length=stint_length,
        gap_to_leader=gap_to_leader,
        gap_to_ahead=gap_to_ahead,
    )


def _make_race_state(
    current_lap: int = 34,
    total_laps: int = 57,
    drivers: list[DriverState] | None = None,
    rainfall: bool = False,
) -> RaceState:
    return RaceState(
        session_key=9158,
        meeting_name="Bahrain GP",
        track_name="Bahrain",
        current_lap=current_lap,
        total_laps=total_laps,
        drivers=drivers or [],
        weather=_make_weather(rainfall=rainfall),
    )


def _make_deg(
    compound: TireCompound = TireCompound.MEDIUM,
    deg_rate: float = 0.08,
    cliff_lap: int | None = 38,
    laps_remaining_estimate: int | None = 4,
    current_stint_laps: int = 18,
) -> TireDegradation:
    return TireDegradation(
        driver_number=1,
        compound=compound,
        deg_rate_per_lap=deg_rate,
        predicted_cliff_lap=cliff_lap,
        laps_remaining_estimate=laps_remaining_estimate,
        current_stint_laps=current_stint_laps,
    )


# ===========================================================================
# Weather tests
# ===========================================================================


class TestWeatherDetection:
    def test_rain_started_event(self):
        history = [_make_weather(rainfall=False), _make_weather(rainfall=True)]
        events = detect_weather_changes(history)
        assert any(e.event == WeatherEvent.RAIN_STARTED for e in events)

    def test_rain_stopped_event(self):
        history = [_make_weather(rainfall=True), _make_weather(rainfall=False)]
        events = detect_weather_changes(history)
        assert any(e.event == WeatherEvent.RAIN_STOPPED for e in events)

    def test_no_events_stable_conditions(self):
        history = [_make_weather() for _ in range(5)]
        events = detect_weather_changes(history)
        assert events == []

    def test_rain_threat_high_humidity_dropping_temp(self):
        history = [
            _make_weather(humidity=82.0, track_temp=40.0),
            _make_weather(humidity=85.0, track_temp=37.0),
            _make_weather(humidity=88.0, track_temp=35.0),
        ]
        events = detect_weather_changes(history)
        assert any(e.event == WeatherEvent.RAIN_THREAT for e in events)

    def test_no_rain_threat_low_humidity(self):
        history = [
            _make_weather(humidity=40.0, track_temp=45.0),
            _make_weather(humidity=42.0, track_temp=42.0),
            _make_weather(humidity=44.0, track_temp=40.0),
        ]
        events = detect_weather_changes(history)
        assert not any(e.event == WeatherEvent.RAIN_THREAT for e in events)

    def test_temp_spike_event(self):
        history = [
            _make_weather(track_temp=38.0),
            _make_weather(track_temp=44.0),  # +6 °C spike
        ]
        events = detect_weather_changes(history)
        assert any(e.event == WeatherEvent.TEMP_SPIKE for e in events)

    def test_wind_shift_event(self):
        history = [
            _make_weather(wind_speed=2.0),
            _make_weather(wind_speed=14.0),  # +12 km/h
        ]
        events = detect_weather_changes(history)
        assert any(e.event == WeatherEvent.WIND_SHIFT for e in events)

    def test_single_reading_returns_no_events(self):
        events = detect_weather_changes([_make_weather()])
        assert events == []

    def test_is_rain_threat_false_stable(self):
        assert not is_rain_threat([_make_weather() for _ in range(3)])

    def test_is_rain_threat_true_converging(self):
        history = [
            _make_weather(humidity=82.0, track_temp=40.0),
            _make_weather(humidity=85.0, track_temp=37.0),
            _make_weather(humidity=88.0, track_temp=35.0),
        ]
        assert is_rain_threat(history)

    def test_conditions_summary_dry(self):
        w = _make_weather(air_temp=28.0, track_temp=42.0, humidity=45.0)
        summary = get_current_conditions_summary(w)
        assert "28.0" in summary
        assert "42.0" in summary

    def test_conditions_summary_rain(self):
        w = _make_weather(rainfall=True)
        summary = get_current_conditions_summary(w)
        assert "RAIN" in summary

    def test_recommend_tire_rain(self):
        w = _make_weather(rainfall=True)
        assert recommend_tire_for_conditions(w, laps_remaining=20) == "INTERMEDIATE"

    def test_recommend_tire_dry_few_laps(self):
        w = _make_weather(rainfall=False)
        assert recommend_tire_for_conditions(w, laps_remaining=8) == "SOFT"


# ===========================================================================
# Strategy: recommend_compound
# ===========================================================================


class TestRecommendCompound:
    def test_rain_always_wet(self):
        w = _make_weather(rainfall=True)
        assert recommend_compound(20, w) == TireCompound.WET

    def test_few_laps_prefers_soft(self):
        w = _make_weather()
        assert recommend_compound(12, w) == TireCompound.SOFT

    def test_medium_range_prefers_medium(self):
        w = _make_weather()
        assert recommend_compound(22, w) == TireCompound.MEDIUM

    def test_many_laps_prefers_hard(self):
        w = _make_weather()
        assert recommend_compound(35, w) == TireCompound.HARD

    def test_respects_available_compounds(self):
        w = _make_weather()
        # Only HARD available even for short stint
        result = recommend_compound(5, w, available_compounds=[TireCompound.HARD])
        assert result == TireCompound.HARD

    def test_available_compounds_no_soft(self):
        w = _make_weather()
        result = recommend_compound(10, w, available_compounds=[TireCompound.MEDIUM, TireCompound.HARD])
        assert result == TireCompound.MEDIUM


# ===========================================================================
# Strategy: calculate_optimal_pit_window
# ===========================================================================


class TestCalculateOptimalPitWindow:
    def test_returns_none_at_race_end(self):
        driver = _make_driver()
        race = _make_race_state(current_lap=57, total_laps=57)
        assert calculate_optimal_pit_window(driver, None, race) is None

    def test_window_with_cliff_data(self):
        driver = _make_driver(compound=TireCompound.MEDIUM, stint_length=18)
        race = _make_race_state(current_lap=34, total_laps=57)
        deg = _make_deg(cliff_lap=38, laps_remaining_estimate=4)
        window = calculate_optimal_pit_window(driver, deg, race)
        assert window is not None
        assert window.earliest_lap <= window.ideal_lap <= window.latest_lap

    def test_window_ideal_before_cliff(self):
        driver = _make_driver(compound=TireCompound.MEDIUM)
        race = _make_race_state(current_lap=30, total_laps=57)
        deg = _make_deg(cliff_lap=38)
        window = calculate_optimal_pit_window(driver, deg, race)
        assert window is not None
        assert window.ideal_lap <= 38

    def test_window_without_deg_data(self):
        driver = _make_driver()
        race = _make_race_state(current_lap=20, total_laps=57)
        window = calculate_optimal_pit_window(driver, None, race)
        assert window is not None
        assert window.earliest_lap < window.latest_lap

    def test_window_bounds_respected(self):
        driver = _make_driver()
        race = _make_race_state(current_lap=44, total_laps=57)
        window = calculate_optimal_pit_window(driver, None, race)
        if window:
            assert window.earliest_lap >= 45  # at least next lap
            assert window.latest_lap <= 57


# ===========================================================================
# Strategy: evaluate_undercut
# ===========================================================================


class TestEvaluateUndercut:
    def _expected_net_loss(self) -> float:
        return DEFAULT_PIT_LOSS_S - FRESH_TIRE_GAIN_PER_LAP_S * FRESH_TIRE_ADVANTAGE_LAPS

    def test_viable_when_gap_small(self):
        driver = _make_driver(number=2, position=2, gap_to_ahead=5.0)
        car_ahead = _make_driver(number=1, position=1)
        viable, _ = evaluate_undercut(driver, car_ahead, None, None)
        assert viable is True

    def test_not_viable_when_gap_large(self):
        driver = _make_driver(number=2, position=2, gap_to_ahead=30.0)
        car_ahead = _make_driver(number=1, position=1)
        viable, _ = evaluate_undercut(driver, car_ahead, None, None)
        assert viable is False

    def test_not_viable_unknown_gap(self):
        driver = _make_driver(number=2, position=2, gap_to_ahead=None)
        car_ahead = _make_driver(number=1, position=1)
        viable, reason = evaluate_undercut(driver, car_ahead, None, None)
        assert viable is False
        assert "unknown" in reason.lower()

    def test_deg_advantage_overrides_marginal_gap(self):
        # Gap is just above viable threshold but driver is degrading much faster
        net_loss = self._expected_net_loss()
        driver = _make_driver(number=2, position=2, gap_to_ahead=net_loss + 0.5)
        car_ahead = _make_driver(number=1, position=1)
        deg_driver = _make_deg(deg_rate=0.25)  # fast degrading
        deg_ahead = _make_deg(deg_rate=0.05)   # slow degrading
        viable, _ = evaluate_undercut(driver, car_ahead, deg_driver, deg_ahead)
        assert viable is True

    def test_reasoning_mentions_driver_name(self):
        driver = _make_driver(number=2, position=2, gap_to_ahead=5.0)
        car_ahead = _make_driver(number=1, position=1)
        car_ahead = DriverState(
            driver_number=1, name="Verstappen", team="RB", position=1,
            tire_compound=TireCompound.HARD,
        )
        _, reason = evaluate_undercut(driver, car_ahead, None, None)
        assert "Verstappen" in reason


# ===========================================================================
# Strategy: evaluate_overcut
# ===========================================================================


class TestEvaluateOvercut:
    def test_viable_when_gap_large(self):
        driver = _make_driver(number=1, position=1)
        car_behind = _make_driver(number=2, position=2, gap_to_ahead=30.0)
        race = _make_race_state()
        viable, _ = evaluate_overcut(driver, car_behind, None, None, race)
        assert viable is True

    def test_risky_when_gap_small(self):
        driver = _make_driver(number=1, position=1)
        car_behind = _make_driver(number=2, position=2, gap_to_ahead=5.0)
        race = _make_race_state()
        viable, _ = evaluate_overcut(driver, car_behind, None, None, race)
        assert viable is False

    def test_viable_due_to_high_deg_behind(self):
        driver = _make_driver(number=1, position=1)
        car_behind = _make_driver(number=2, position=2, gap_to_ahead=15.0)
        race = _make_race_state()
        deg_behind = _make_deg(deg_rate=0.35)  # very high
        viable, reason = evaluate_overcut(driver, car_behind, None, deg_behind, race)
        assert viable is True
        assert "degrading" in reason.lower()

    def test_not_viable_unknown_gap(self):
        driver = _make_driver(number=1, position=1)
        car_behind = _make_driver(number=2, position=2, gap_to_ahead=None)
        race = _make_race_state()
        viable, reason = evaluate_overcut(driver, car_behind, None, None, race)
        assert viable is False
        assert "unknown" in reason.lower()


# ===========================================================================
# Strategy: build_strategy_recommendation
# ===========================================================================


class TestBuildStrategyRecommendation:
    def test_pit_now_on_rain(self):
        driver = _make_driver(compound=TireCompound.MEDIUM, stint_length=10)
        race = _make_race_state(rainfall=True, drivers=[driver])
        rec = build_strategy_recommendation(driver, race, None, [_make_weather(rainfall=True)])
        assert rec.recommended_action == "PIT_NOW"
        assert rec.confidence >= 0.9

    def test_pit_now_cliff_imminent(self):
        driver = _make_driver(compound=TireCompound.MEDIUM, stint_length=25)
        race = _make_race_state(current_lap=36, total_laps=57, drivers=[driver])
        deg = _make_deg(cliff_lap=38, laps_remaining_estimate=1)
        rec = build_strategy_recommendation(driver, race, deg, [_make_weather()])
        assert rec.recommended_action == "PIT_NOW"

    def test_stay_out_early_race(self):
        driver = _make_driver(compound=TireCompound.MEDIUM, stint_length=3)
        race = _make_race_state(current_lap=5, total_laps=57, drivers=[driver])
        deg = _make_deg(cliff_lap=30, laps_remaining_estimate=25)
        rec = build_strategy_recommendation(driver, race, deg, [_make_weather()])
        assert rec.recommended_action == "STAY_OUT"

    def test_recommendation_has_compound(self):
        driver = _make_driver(compound=TireCompound.MEDIUM, stint_length=18)
        race = _make_race_state(drivers=[driver])
        rec = build_strategy_recommendation(driver, race, None, [_make_weather()])
        assert rec.recommended_compound is not None
        assert isinstance(rec.recommended_compound, TireCompound)

    def test_confidence_between_0_and_1(self):
        driver = _make_driver()
        race = _make_race_state(drivers=[driver])
        rec = build_strategy_recommendation(driver, race, None, [_make_weather()])
        assert 0.0 <= rec.confidence <= 1.0

    def test_undercut_flag_set_when_viable(self):
        # Driver 2 is 5 s behind leader — undercut should be viable
        driver = _make_driver(number=2, position=2, gap_to_ahead=5.0)
        leader = _make_driver(number=1, position=1)
        race = _make_race_state(drivers=[leader, driver])
        rec = build_strategy_recommendation(driver, race, None, [_make_weather()])
        assert rec.undercut_viable is True

    def test_reasoning_is_non_empty(self):
        driver = _make_driver()
        race = _make_race_state(drivers=[driver])
        rec = build_strategy_recommendation(driver, race, None, [_make_weather()])
        assert len(rec.reasoning) > 0

    def test_rain_threat_lowers_confidence(self):
        driver = _make_driver()
        race = _make_race_state(drivers=[driver])
        # Build converging weather history
        weather_history = [
            _make_weather(humidity=82.0, track_temp=40.0),
            _make_weather(humidity=86.0, track_temp=37.0),
            _make_weather(humidity=90.0, track_temp=34.0),
        ]
        rec = build_strategy_recommendation(driver, race, None, weather_history)
        assert rec.confidence <= 0.6

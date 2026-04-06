"""Unit tests for race_builder — exercises model construction from raw data."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.models import TireCompound
from src.data.openf1_client import OpenF1Client
from src.data.race_builder import (
    _build_lap,
    _build_pit,
    _build_stint,
    _build_weather,
    _parse_gap,
    build_race_state,
)


# ---------------------------------------------------------------------------
# Unit: sub-model builders
# ---------------------------------------------------------------------------


def test_build_lap_normal():
    raw = {
        "lap_number": 5,
        "lap_duration": 95.321,
        "duration_sector_1": 31.1,
        "duration_sector_2": 32.2,
        "duration_sector_3": 32.021,
        "is_pit_in_lap": False,
        "is_pit_out_lap": False,
    }
    lap = _build_lap(raw)
    assert lap.lap_number == 5
    assert lap.lap_time == pytest.approx(95.321)
    assert lap.sector_times.sector_1 == pytest.approx(31.1)
    assert not lap.is_pit_in_lap


def test_build_lap_pit_in():
    raw = {
        "lap_number": 27,
        "lap_duration": None,
        "duration_sector_1": None,
        "duration_sector_2": None,
        "duration_sector_3": None,
        "is_pit_in_lap": True,
        "is_pit_out_lap": False,
    }
    lap = _build_lap(raw)
    assert lap.is_pit_in_lap is True
    assert lap.lap_time is None


def test_build_lap_missing_sectors():
    raw = {"lap_number": 3, "lap_duration": 96.0}
    lap = _build_lap(raw)
    assert lap.sector_times is None
    assert lap.lap_time == 96.0


def test_build_stint_compound_mapping():
    raw = {"compound": "medium", "lap_start": 1, "lap_end": 27, "tyre_age_at_start": 0}
    stint = _build_stint(raw, number=1)
    assert stint.compound == TireCompound.MEDIUM
    assert stint.lap_start == 1
    assert stint.stint_number == 1


def test_build_stint_unknown_compound():
    raw = {"compound": "HYPERSOFT", "lap_start": 1, "lap_end": 15}
    stint = _build_stint(raw, number=1)
    assert stint.compound == TireCompound.UNKNOWN


def test_build_pit(raw_stints):
    raw = {"driver_number": 1, "lap_number": 27, "pit_duration": 22.3}
    pit = _build_pit(raw, raw_stints)
    assert pit.lap_number == 27
    assert pit.stop_duration == pytest.approx(22.3)


def test_build_weather(raw_weather):
    weather = _build_weather(raw_weather, current_lap=30, laps_raw=[])
    assert weather.air_temp == pytest.approx(28.4)
    assert weather.track_temp == pytest.approx(42.1)
    assert weather.rainfall is False


def test_build_weather_empty():
    from src.data.race_builder import _build_weather
    weather = _build_weather([], current_lap=1, laps_raw=[])
    assert weather.air_temp is None


# ---------------------------------------------------------------------------
# Unit: _parse_gap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("+4.231", 4.231),
        ("0.000", 0.0),
        ("+0.000", 0.0),
        ("1 LAP", None),
        ("DNF", None),
        (None, None),
        (4.5, 4.5),
    ],
)
def test_parse_gap(raw, expected):
    result = _parse_gap(raw)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Integration: build_race_state with mocked client
# ---------------------------------------------------------------------------


def _make_mock_client(
    sessions, drivers, laps, stints, pits, intervals, weather, rc
) -> OpenF1Client:
    client = MagicMock(spec=OpenF1Client)
    client.get_sessions = AsyncMock(return_value=sessions)
    client.get_drivers = AsyncMock(return_value=drivers)
    client.get_laps = AsyncMock(return_value=laps)
    client.get_stints = AsyncMock(return_value=stints)
    client.get_pit = AsyncMock(return_value=pits)
    client.get_intervals = AsyncMock(return_value=intervals)
    client.get_weather = AsyncMock(return_value=weather)
    client.get_race_control = AsyncMock(return_value=rc)
    return client


@pytest.mark.asyncio
async def test_build_race_state_basic(
    raw_sessions,
    raw_drivers,
    raw_laps_verstappen,
    raw_stints,
    raw_pits,
    raw_intervals,
    raw_weather,
    raw_race_control,
):
    # Only VER has laps in this fixture; NOR and LEC will have empty lap lists
    client = _make_mock_client(
        raw_sessions,
        raw_drivers,
        raw_laps_verstappen,  # all laps (only driver 1)
        raw_stints,
        raw_pits,
        raw_intervals,
        raw_weather,
        raw_race_control,
    )

    state = await build_race_state(client, session_key=9158)

    assert state.session_key == 9158
    assert state.meeting_name == "Bahrain Grand Prix"
    assert state.total_laps == 57
    assert len(state.drivers) == 3


@pytest.mark.asyncio
async def test_build_race_state_driver_tire_compound(
    raw_sessions,
    raw_drivers,
    raw_laps_verstappen,
    raw_stints,
    raw_pits,
    raw_intervals,
    raw_weather,
    raw_race_control,
):
    client = _make_mock_client(
        raw_sessions, raw_drivers, raw_laps_verstappen,
        raw_stints, raw_pits, raw_intervals, raw_weather, raw_race_control,
    )
    state = await build_race_state(client, session_key=9158)

    ver = next(d for d in state.drivers if d.driver_number == 1)
    # Last stint for VER is HARD
    assert ver.tire_compound == TireCompound.HARD
    assert len(ver.stints) == 2
    assert len(ver.pit_stops) == 1


@pytest.mark.asyncio
async def test_build_race_state_weather(
    raw_sessions, raw_drivers, raw_laps_verstappen, raw_stints,
    raw_pits, raw_intervals, raw_weather, raw_race_control,
):
    client = _make_mock_client(
        raw_sessions, raw_drivers, raw_laps_verstappen,
        raw_stints, raw_pits, raw_intervals, raw_weather, raw_race_control,
    )
    state = await build_race_state(client, session_key=9158)

    assert state.weather.air_temp == pytest.approx(28.4)
    assert state.weather.rainfall is False


@pytest.mark.asyncio
async def test_build_race_state_replay_mode(
    raw_sessions, raw_drivers, raw_laps_verstappen, raw_stints,
    raw_pits, raw_intervals, raw_weather, raw_race_control,
):
    client = _make_mock_client(
        raw_sessions, raw_drivers, raw_laps_verstappen,
        raw_stints, raw_pits, raw_intervals, raw_weather, raw_race_control,
    )
    state = await build_race_state(client, session_key=9158, up_to_lap=20)

    # current_lap should be capped at 20 (laps are filtered before _max_lap)
    assert state.current_lap == 20


@pytest.mark.asyncio
async def test_build_race_state_missing_session_raises(raw_drivers):
    client = _make_mock_client(
        [], raw_drivers, [], [], [], [], [], []
    )
    with pytest.raises(ValueError, match="No session found"):
        await build_race_state(client, session_key=9999)


@pytest.mark.asyncio
async def test_build_race_state_race_control_filter(
    raw_sessions, raw_drivers, raw_laps_verstappen, raw_stints,
    raw_pits, raw_intervals, raw_weather, raw_race_control,
):
    """RC messages after up_to_lap should be excluded in replay mode."""
    client = _make_mock_client(
        raw_sessions, raw_drivers, raw_laps_verstappen,
        raw_stints, raw_pits, raw_intervals, raw_weather, raw_race_control,
    )
    # The VSC message is at lap 30 — should be excluded when up_to_lap=20
    state = await build_race_state(client, session_key=9158, up_to_lap=20)
    vsc_messages = [m for m in state.race_control if "SAFETY CAR" in m.message]
    assert len(vsc_messages) == 0

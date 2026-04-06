"""Shared pytest fixtures for Pitwall-AI tests."""

import pytest

# ---------------------------------------------------------------------------
# Raw OpenF1 API response fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_sessions() -> list[dict]:
    return [
        {
            "session_key": 9158,
            "meeting_name": "Bahrain Grand Prix",
            "location": "Bahrain International Circuit",
            "country_name": "Bahrain",
            "session_type": "Race",
            "year": 2024,
            "total_laps": 57,
            "session_status": "Finished",
        }
    ]


@pytest.fixture()
def raw_drivers() -> list[dict]:
    return [
        {
            "driver_number": 1,
            "full_name": "Max Verstappen",
            "broadcast_name": "VER",
            "team_name": "Red Bull Racing",
            "session_key": 9158,
        },
        {
            "driver_number": 4,
            "full_name": "Lando Norris",
            "broadcast_name": "NOR",
            "team_name": "McLaren",
            "session_key": 9158,
        },
        {
            "driver_number": 16,
            "full_name": "Charles Leclerc",
            "broadcast_name": "LEC",
            "team_name": "Ferrari",
            "session_key": 9158,
        },
    ]


@pytest.fixture()
def raw_stints() -> list[dict]:
    return [
        # Verstappen: Medium start → Hard finish
        {"driver_number": 1, "stint_number": 1, "compound": "MEDIUM", "lap_start": 1, "lap_end": 27, "tyre_age_at_start": 0},
        {"driver_number": 1, "stint_number": 2, "compound": "HARD", "lap_start": 28, "lap_end": 57, "tyre_age_at_start": 0},
        # Norris: Soft start → Hard finish
        {"driver_number": 4, "stint_number": 1, "compound": "SOFT", "lap_start": 1, "lap_end": 18, "tyre_age_at_start": 0},
        {"driver_number": 4, "stint_number": 2, "compound": "HARD", "lap_start": 19, "lap_end": 57, "tyre_age_at_start": 0},
        # Leclerc: Medium → Hard → Hard
        {"driver_number": 16, "stint_number": 1, "compound": "MEDIUM", "lap_start": 1, "lap_end": 20, "tyre_age_at_start": 0},
        {"driver_number": 16, "stint_number": 2, "compound": "HARD", "lap_start": 21, "lap_end": 40, "tyre_age_at_start": 0},
        {"driver_number": 16, "stint_number": 3, "compound": "HARD", "lap_start": 41, "lap_end": 57, "tyre_age_at_start": 0},
    ]


@pytest.fixture()
def raw_pits() -> list[dict]:
    return [
        {"driver_number": 1, "lap_number": 27, "pit_duration": 22.3, "session_key": 9158},
        {"driver_number": 4, "lap_number": 18, "pit_duration": 21.8, "session_key": 9158},
        {"driver_number": 16, "lap_number": 20, "pit_duration": 23.1, "session_key": 9158},
        {"driver_number": 16, "lap_number": 40, "pit_duration": 22.7, "session_key": 9158},
    ]


@pytest.fixture()
def raw_laps_verstappen() -> list[dict]:
    """27 clean medium laps + 1 pit-in + 29 hard laps."""
    laps = []
    base_time = 95.5  # seconds
    for i in range(1, 58):
        is_pit_in = i == 27
        is_pit_out = i == 28
        # Simulate mild degradation on mediums, then fresh pace on hards
        if i <= 27:
            t = base_time + (i - 1) * 0.05 if not is_pit_in else None
        else:
            t = base_time - 0.3 + (i - 28) * 0.04 if not is_pit_out else None
        laps.append({
            "driver_number": 1,
            "lap_number": i,
            "lap_duration": t,
            "duration_sector_1": round(t / 3, 3) if t else None,
            "duration_sector_2": round(t / 3, 3) if t else None,
            "duration_sector_3": round(t / 3, 3) if t else None,
            "is_pit_in_lap": is_pit_in,
            "is_pit_out_lap": is_pit_out,
            "session_key": 9158,
        })
    return laps


@pytest.fixture()
def raw_intervals() -> list[dict]:
    return [
        {"driver_number": 1, "gap_to_leader": "+0.000", "interval": "+0.000", "session_key": 9158},
        {"driver_number": 4, "gap_to_leader": "+4.231", "interval": "+4.231", "session_key": 9158},
        {"driver_number": 16, "gap_to_leader": "+8.015", "interval": "+3.784", "session_key": 9158},
    ]


@pytest.fixture()
def raw_weather() -> list[dict]:
    return [
        {
            "air_temperature": 28.4,
            "track_temperature": 42.1,
            "humidity": 44.5,
            "rainfall": False,
            "wind_speed": 3.2,
            "wind_direction": 180,
            "session_key": 9158,
        }
    ]


@pytest.fixture()
def raw_race_control() -> list[dict]:
    return [
        {
            "date": "2024-03-02T15:01:00+00:00",
            "message": "GREEN LIGHT - PIT EXIT OPEN",
            "flag": "GREEN",
            "category": "Flag",
            "lap_number": 1,
            "session_key": 9158,
        },
        {
            "date": "2024-03-02T15:45:00+00:00",
            "message": "VIRTUAL SAFETY CAR DEPLOYED",
            "flag": None,
            "category": "SafetyCar",
            "lap_number": 30,
            "session_key": 9158,
        },
    ]

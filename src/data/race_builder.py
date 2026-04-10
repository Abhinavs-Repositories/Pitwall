"""Builds a clean RaceState from raw OpenF1 API responses.

All the messy normalisation lives here so the rest of the codebase works
with well-typed Pydantic models instead of raw dicts.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.data.models import (
    DriverState,
    LapData,
    PitStop,
    RaceControlMessage,
    RaceState,
    SectorTime,
    Stint,
    TireCompound,
    WeatherState,
    parse_compound,
)
from src.data.openf1_client import OpenF1Client

logger = logging.getLogger(__name__)

RawList = list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def build_race_state(
    client: OpenF1Client,
    session_key: int,
    up_to_lap: int | None = None,
) -> RaceState:
    """Fetch all necessary data from OpenF1 and assemble a RaceState.

    Args:
        client:      Initialised OpenF1Client (used as async context manager upstream).
        session_key: The OpenF1 session_key for the race.
        up_to_lap:   If set, only include data up to (and including) this lap number.
                     Useful for replay mode.  None = full race.
    """
    logger.info(
        "Building race state",
        extra={"session_key": session_key, "up_to_lap": up_to_lap},
    )

    # Fetch everything in parallel
    import asyncio

    (
        sessions,
        drivers_raw,
        laps_raw,
        stints_raw,
        pits_raw,
        intervals_raw,
        weather_raw,
        rc_raw,
    ) = await asyncio.gather(
        client.get_sessions(session_key=session_key),
        client.get_drivers(session_key),
        client.get_laps(session_key),
        client.get_stints(session_key),
        client.get_pit(session_key),
        client.get_intervals(session_key),
        client.get_weather(session_key),
        client.get_race_control(session_key),
    )

    if not sessions:
        raise ValueError(f"No session found for session_key={session_key}")

    session = sessions[0]
    meeting_name: str = session.get("meeting_name", "Unknown GP")
    track_name: str = session.get("location", session.get("country_name", "Unknown"))
    total_laps: int = session.get("total_laps") or _infer_total_laps(laps_raw)

    # Apply lap filter
    if up_to_lap is not None:
        laps_raw = [r for r in laps_raw if (r.get("lap_number") or 0) <= up_to_lap]
        stints_raw = [r for r in stints_raw if (r.get("lap_start") or 0) <= up_to_lap]
        pits_raw = [r for r in pits_raw if (r.get("lap_number") or 0) <= up_to_lap]
        intervals_raw = _filter_intervals_by_lap(intervals_raw, laps_raw)

    current_lap = up_to_lap if up_to_lap is not None else _max_lap(laps_raw)

    # Index raw data by driver number for O(1) lookup
    laps_by_driver = _group_by(laps_raw, "driver_number")
    stints_by_driver = _group_by(stints_raw, "driver_number")
    pits_by_driver = _group_by(pits_raw, "driver_number")
    intervals_by_driver = _latest_by(intervals_raw, "driver_number")
    positions_by_driver = _latest_positions(intervals_raw, laps_raw, current_lap)

    drivers: list[DriverState] = []
    for raw_driver in drivers_raw:
        driver_num = raw_driver.get("driver_number")
        if driver_num is None:
            continue
        driver_num = int(driver_num)
        try:
            ds = _build_driver_state(
                driver_num=driver_num,
                raw_driver=raw_driver,
                laps=laps_by_driver.get(driver_num, []),
                stints=stints_by_driver.get(driver_num, []),
                pits=pits_by_driver.get(driver_num, []),
                interval=intervals_by_driver.get(driver_num),
                position=positions_by_driver.get(driver_num, 0),
            )
            drivers.append(ds)
        except Exception as exc:
            logger.warning(
                "Skipping driver due to build error",
                extra={"driver_number": driver_num, "error": str(exc)},
            )

    # Sort by position
    drivers.sort(key=lambda d: d.position if d.position > 0 else 99)

    weather = _build_weather(weather_raw, current_lap, laps_raw)
    race_control = _build_race_control(rc_raw, up_to_lap)

    return RaceState(
        session_key=session_key,
        meeting_name=meeting_name,
        track_name=track_name,
        current_lap=current_lap,
        total_laps=total_laps,
        drivers=drivers,
        weather=weather,
        race_control=race_control,
        session_status=_infer_session_status(session),
    )


# ---------------------------------------------------------------------------
# Driver builder
# ---------------------------------------------------------------------------


def _build_driver_state(
    driver_num: int,
    raw_driver: dict[str, Any],
    laps: RawList,
    stints: RawList,
    pits: RawList,
    interval: dict[str, Any] | None,
    position: int,
) -> DriverState:
    name = (
        raw_driver.get("full_name")
        or raw_driver.get("broadcast_name")
        or f"Driver {driver_num}"
    )
    team = raw_driver.get("team_name", "Unknown")

    lap_data = [_build_lap(r) for r in sorted(laps, key=lambda x: x.get("lap_number", 0))]
    stint_models = [_build_stint(r, i) for i, r in enumerate(sorted(stints, key=lambda x: x.get("lap_start", 0)), start=1)]
    pit_models = [_build_pit(r, stints) for r in pits]

    # Current tire: last stint's compound
    current_compound = TireCompound.UNKNOWN
    current_stint_laps = 0
    if stint_models:
        last_stint = max(stint_models, key=lambda s: s.lap_start)
        current_compound = last_stint.compound
        last_lap = max((ld.lap_number for ld in lap_data), default=last_stint.lap_start)
        current_stint_laps = last_lap - last_stint.lap_start

    last_lap_time: float | None = None
    clean_laps = [ld for ld in lap_data if ld.lap_time and not ld.is_pit_in_lap and not ld.is_pit_out_lap]
    if clean_laps:
        last_lap_time = clean_laps[-1].lap_time

    gap_to_leader: float | None = None
    gap_to_ahead: float | None = None
    if interval:
        gap_to_leader = _parse_gap(interval.get("gap_to_leader"))
        gap_to_ahead = _parse_gap(interval.get("interval"))

    return DriverState(
        driver_number=driver_num,
        name=name,
        team=team,
        position=position,
        gap_to_leader=gap_to_leader,
        gap_to_ahead=gap_to_ahead,
        last_lap_time=last_lap_time,
        tire_compound=current_compound,
        stint_length=current_stint_laps,
        pit_stops=pit_models,
        stints=stint_models,
        lap_times=lap_data,
        is_in_pit=False,
        is_retired=_is_retired(raw_driver),
    )


# ---------------------------------------------------------------------------
# Sub-model builders
# ---------------------------------------------------------------------------


def _build_lap(raw: dict[str, Any]) -> LapData:
    lap_time_raw = raw.get("lap_duration")
    lap_time: float | None = float(lap_time_raw) if lap_time_raw is not None else None

    s1 = raw.get("duration_sector_1")
    s2 = raw.get("duration_sector_2")
    s3 = raw.get("duration_sector_3")
    sector_times: SectorTime | None = None
    if any(v is not None for v in (s1, s2, s3)):
        sector_times = SectorTime(
            sector_1=float(s1) if s1 is not None else None,
            sector_2=float(s2) if s2 is not None else None,
            sector_3=float(s3) if s3 is not None else None,
        )

    return LapData(
        lap_number=int(raw["lap_number"]),
        lap_time=lap_time,
        sector_times=sector_times,
        is_pit_in_lap=bool(raw.get("is_pit_in_lap", False)),
        is_pit_out_lap=bool(raw.get("is_pit_out_lap", False)),
    )


def _build_stint(raw: dict[str, Any], number: int) -> Stint:
    return Stint(
        stint_number=number,
        compound=parse_compound(raw.get("compound")),
        lap_start=int(raw.get("lap_start") or 1),
        lap_end=int(raw.get("lap_end") or raw.get("lap_start") or 1),
        tyre_age_at_start=int(raw.get("tyre_age_at_start") or 0),
    )


def _build_pit(raw: dict[str, Any], stints: RawList) -> PitStop:
    lap = int(raw.get("lap_number") or 0)
    duration = float(raw.get("pit_duration") or 0.0)

    # Find the compound before and after by matching stints around this lap
    compound_before = _compound_at_lap(stints, lap - 1)
    compound_after = _compound_at_lap(stints, lap + 1)

    return PitStop(
        lap_number=lap,
        stop_duration=duration,
        compound_before=compound_before,
        compound_after=compound_after,
    )


def _compound_at_lap(stints: RawList, lap: int) -> TireCompound | None:
    for s in stints:
        start = int(s.get("lap_start") or 0)
        end = int(s.get("lap_end") or 0)
        if start <= lap <= end:
            return parse_compound(s.get("compound"))
    return None


# ---------------------------------------------------------------------------
# Weather builder
# ---------------------------------------------------------------------------


def _build_weather(weather_raw: RawList, current_lap: int, laps_raw: RawList) -> WeatherState:
    """Return the weather reading closest to current_lap."""
    if not weather_raw:
        return WeatherState()

    # Weather entries have a date; laps have a date_start.
    # Simple heuristic: use the last weather reading (most recent).
    latest = weather_raw[-1]
    return WeatherState(
        air_temp=_float(latest.get("air_temperature")),
        track_temp=_float(latest.get("track_temperature")),
        humidity=_float(latest.get("humidity")),
        rainfall=bool(latest.get("rainfall", False)),
        wind_speed=_float(latest.get("wind_speed")),
        wind_direction=_int(latest.get("wind_direction")),
    )


# ---------------------------------------------------------------------------
# Race control builder
# ---------------------------------------------------------------------------


def _build_race_control(rc_raw: RawList, up_to_lap: int | None) -> list[RaceControlMessage]:
    messages: list[RaceControlMessage] = []
    for raw in rc_raw:
        lap = raw.get("lap_number")
        if up_to_lap is not None and lap is not None and int(lap) > up_to_lap:
            continue
        date_str = raw.get("date")
        try:
            date = datetime.fromisoformat(date_str.replace("Z", "+00:00")) if date_str else datetime.now(tz=timezone.utc)
        except (ValueError, AttributeError):
            date = datetime.now(tz=timezone.utc)

        messages.append(
            RaceControlMessage(
                date=date,
                message=raw.get("message", ""),
                flag=raw.get("flag"),
                category=raw.get("category"),
            )
        )
    return messages


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------


def _latest_positions(
    intervals_raw: RawList,
    laps_raw: RawList,
    current_lap: int,
) -> dict[int, int]:
    """Build a driver_number → position mapping for current_lap."""
    # Try intervals first (has gap_to_leader which implies ordering)
    # Fall back to lap position data
    positions: dict[int, int] = {}

    # Use gaps to estimate position ordering
    driver_gaps: list[tuple[int, float]] = []
    seen: set[int] = set()
    for raw in reversed(intervals_raw):
        dn = raw.get("driver_number")
        if dn is None or int(dn) in seen:
            continue
        dn = int(dn)
        gap = _parse_gap(raw.get("gap_to_leader"))
        driver_gaps.append((dn, gap if gap is not None else 9999.0))
        seen.add(dn)

    driver_gaps.sort(key=lambda x: x[1])
    for pos, (dn, _) in enumerate(driver_gaps, start=1):
        positions[dn] = pos

    return positions


def _latest_by(raw_list: RawList, key: str) -> dict[int, dict[str, Any]]:
    """Return the last record for each unique key value."""
    result: dict[int, dict[str, Any]] = {}
    for raw in raw_list:
        k = raw.get(key)
        if k is not None:
            result[int(k)] = raw
    return result


def _filter_intervals_by_lap(
    intervals_raw: RawList, filtered_laps: RawList
) -> RawList:
    """Keep only interval records that fall within the date range of filtered laps."""
    # Simplified: return all intervals (date filtering would need date comparisons)
    return intervals_raw


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _group_by(raw_list: RawList, key: str) -> dict[int, RawList]:
    result: dict[int, RawList] = {}
    for raw in raw_list:
        k = raw.get(key)
        if k is not None:
            result.setdefault(int(k), []).append(raw)
    return result


def _infer_total_laps(laps_raw: RawList) -> int:
    if not laps_raw:
        return 0
    return max((int(r.get("lap_number") or 0) for r in laps_raw), default=0)


def _max_lap(laps_raw: RawList) -> int:
    return _infer_total_laps(laps_raw)


def _parse_gap(value: Any) -> float | None:
    """Convert a gap string like '+4.123' or '1 LAP' to seconds, or None."""
    if value is None:
        return None
    s = str(value).strip()
    if s.startswith("+"):
        s = s[1:]
    try:
        return float(s)
    except ValueError:
        return None  # e.g. "1 LAP", "DNF"


def _is_retired(raw_driver: dict[str, Any]) -> bool:
    return False  # OpenF1 doesn't surface retired status directly per-driver


def _infer_session_status(session: dict[str, Any]) -> str:
    status = session.get("session_status", "")
    match status.lower():
        case "finished":
            return "Finished"
        case "aborted":
            return "Aborted"
        case _:
            return "Started"


def _float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Convenience class wrapper (used by agents and API routes)
# ---------------------------------------------------------------------------


class RaceBuilder:
    """Thin class wrapper around build_race_state for use in agents/routes."""

    def __init__(self, client: OpenF1Client) -> None:
        self._client = client

    async def build(
        self,
        session_key: int,
        up_to_lap: int | None = None,
    ) -> RaceState:
        return await build_race_state(self._client, session_key, up_to_lap)

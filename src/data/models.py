"""Pydantic data models for Pitwall-AI."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TireCompound(str, Enum):
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"
    UNKNOWN = "UNKNOWN"


# Map OpenF1 compound strings → TireCompound
_COMPOUND_MAP: dict[str, TireCompound] = {
    "soft": TireCompound.SOFT,
    "medium": TireCompound.MEDIUM,
    "hard": TireCompound.HARD,
    "intermediate": TireCompound.INTERMEDIATE,
    "wet": TireCompound.WET,
}


def parse_compound(raw: str | None) -> TireCompound:
    """Convert a raw OpenF1 compound string to TireCompound enum."""
    if not raw:
        return TireCompound.UNKNOWN
    return _COMPOUND_MAP.get(raw.lower(), TireCompound.UNKNOWN)


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class SectorTime(BaseModel):
    sector_1: Optional[float] = None
    sector_2: Optional[float] = None
    sector_3: Optional[float] = None


class PitStop(BaseModel):
    lap_number: int
    stop_duration: float
    compound_before: Optional[TireCompound] = None
    compound_after: Optional[TireCompound] = None


class Stint(BaseModel):
    stint_number: int
    compound: TireCompound
    lap_start: int
    lap_end: int
    tyre_age_at_start: int = 0


class LapData(BaseModel):
    lap_number: int
    lap_time: Optional[float] = None  # seconds
    sector_times: Optional[SectorTime] = None
    is_pit_in_lap: bool = False
    is_pit_out_lap: bool = False


# ---------------------------------------------------------------------------
# Core race state models
# ---------------------------------------------------------------------------


class DriverState(BaseModel):
    driver_number: int
    name: str
    team: str
    position: int
    gap_to_leader: Optional[float] = None
    gap_to_ahead: Optional[float] = None
    last_lap_time: Optional[float] = None
    tire_compound: TireCompound
    stint_length: int = 0  # laps on current tires
    pit_stops: list[PitStop] = Field(default_factory=list)
    stints: list[Stint] = Field(default_factory=list)
    lap_times: list[LapData] = Field(default_factory=list)
    is_in_pit: bool = False
    is_retired: bool = False


class WeatherState(BaseModel):
    air_temp: Optional[float] = None
    track_temp: Optional[float] = None
    humidity: Optional[float] = None
    rainfall: bool = False
    wind_speed: Optional[float] = None
    wind_direction: Optional[int] = None


class RaceControlMessage(BaseModel):
    date: datetime
    message: str
    flag: Optional[str] = None      # YELLOW, RED, GREEN, etc.
    category: Optional[str] = None  # SafetyCar, Flag, etc.


class RaceState(BaseModel):
    session_key: int
    meeting_name: str
    track_name: str
    current_lap: int
    total_laps: int
    drivers: list[DriverState] = Field(default_factory=list)
    weather: WeatherState = Field(default_factory=WeatherState)
    race_control: list[RaceControlMessage] = Field(default_factory=list)
    session_status: str = "Started"  # Started, Finished, Aborted


# ---------------------------------------------------------------------------
# Analysis models
# ---------------------------------------------------------------------------


class TireDegradation(BaseModel):
    driver_number: int
    compound: TireCompound
    deg_rate_per_lap: float  # seconds lost per lap
    predicted_cliff_lap: Optional[int] = None
    laps_remaining_estimate: Optional[int] = None
    current_stint_laps: int


class StrategyRecommendation(BaseModel):
    driver_number: int
    recommended_action: str  # "PIT_NOW" | "STAY_OUT" | "PIT_IN_X_LAPS"
    recommended_compound: Optional[TireCompound] = None
    optimal_pit_window: Optional[tuple[int, int]] = None  # (earliest_lap, latest_lap)
    undercut_viable: bool = False
    overcut_viable: bool = False
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# RAG / knowledge base models
# ---------------------------------------------------------------------------


class HistoricalStrategy(BaseModel):
    """Document stored in the Qdrant vector knowledge base."""

    race_name: str
    year: int
    track: str
    winner: str
    winner_strategy: str  # e.g. "M-H-H"
    total_laps: int
    pit_stops_winner: int
    weather_conditions: str
    key_events: str          # safety cars, rain, etc.
    summary: str             # LLM-generated natural language summary


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    session_key: int
    current_lap: int
    message: str
    conversation_history: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    strategy_data: Optional[StrategyRecommendation] = None
    agents_used: list[str]
    processing_time_ms: float

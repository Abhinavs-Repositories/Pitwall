// Mirrors src/data/models.py exactly — keep in sync with the backend.

export type TireCompound = "SOFT" | "MEDIUM" | "HARD" | "INTERMEDIATE" | "WET" | "UNKNOWN";

export interface SectorTime {
  sector_1: number | null;
  sector_2: number | null;
  sector_3: number | null;
}

export interface PitStop {
  lap_number: number;
  stop_duration: number;
  compound_before: TireCompound | null;
  compound_after: TireCompound | null;
}

export interface Stint {
  stint_number: number;
  compound: TireCompound;
  lap_start: number;
  lap_end: number;
  tyre_age_at_start: number;
}

export interface LapData {
  lap_number: number;
  lap_time: number | null;
  sector_times: SectorTime | null;
  is_pit_in_lap: boolean;
  is_pit_out_lap: boolean;
}

export interface DriverState {
  driver_number: number;
  name: string;
  team: string;
  position: number;
  gap_to_leader: number | null;
  gap_to_ahead: number | null;
  last_lap_time: number | null;
  tire_compound: TireCompound;
  stint_length: number;
  pit_stops: PitStop[];
  stints: Stint[];
  lap_times: LapData[];
  is_in_pit: boolean;
  is_retired: boolean;
}

export interface WeatherState {
  air_temp: number | null;
  track_temp: number | null;
  humidity: number | null;
  rainfall: boolean;
  wind_speed: number | null;
  wind_direction: number | null;
}

export interface RaceControlMessage {
  date: string;
  message: string;
  flag: string | null;
  category: string | null;
}

export interface RaceState {
  session_key: number;
  meeting_name: string;
  track_name: string;
  current_lap: number;
  total_laps: number;
  drivers: DriverState[];
  weather: WeatherState;
  race_control: RaceControlMessage[];
  session_status: string;
}

export interface StrategyRecommendation {
  driver_number: number;
  recommended_action: string; // "PIT_NOW" | "STAY_OUT" | "PIT_IN_X_LAPS" (formatted string)
  recommended_compound: TireCompound | null;
  optimal_pit_window: [number, number] | null;
  undercut_viable: boolean;
  overcut_viable: boolean;
  reasoning: string;
  confidence: number; // 0..1
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  session_key: number;
  current_lap: number;
  message: string;
  conversation_history: ChatMessage[];
}

export interface ChatResponse {
  response: string;
  strategy_data: StrategyRecommendation | null;
  agents_used: string[];
  processing_time_ms: number;
}

// Raw OpenF1 session object as returned by GET /api/races (not strongly
// typed server-side — this is the subset the UI actually reads).
export interface RaceSummaryListing {
  session_key: number;
  session_name: string;
  date_start: string;
  circuit_short_name: string;
  country_name: string;
  location: string;
  year: number;
}

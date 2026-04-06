# PITWALL-AI: AI-Powered F1 Race Strategy Agent

## Project Overview

An AI-powered F1 race engineer that analyzes historical and live race data and recommends pit strategy using multi-agent orchestration. Users can select any race from 2023-2025, view live standings, get auto-generated race summaries, and ask natural language questions about race strategy.

**One-liner:** "Should Verstappen pit now?" — answered with data, not guesswork.

**Why this exists:** F1 fans constantly wonder about pit strategy, tire degradation, and undercuts during races. No tool currently lets fans ask these questions and get data-backed answers in natural language. Pitwall-AI is that tool.

---

## Hard Constraints

- **ZERO COST** — Every tool, API, and service must be free. No paid tiers, no credit cards.
- **Heavy backend / light frontend** — This project showcases AI/backend engineering, not React skills.
- **Production patterns** — Code should follow production best practices (typing, error handling, logging, tests) even though it's a personal project. This will be reviewed by hiring managers.

---

## Tech Stack (All Free)

| Component | Tool | Details |
|---|---|---|
| **LLM (Primary)** | Groq API (Llama 3.3 70B) | Free tier, fast inference, OpenAI-compatible SDK. No credit card needed. |
| **LLM (Fallback)** | Google Gemini 2.5 Flash via AI Studio | Free tier, 1M context window. Use when Groq hits rate limits. |
| **F1 Data** | OpenF1 API (free tier) | 18 endpoints, all historical data since 2023, JSON/CSV, no auth needed. Base URL: `https://api.openf1.org/v1`. Rate limit: 3 req/s, 30 req/min. |
| **Vector DB** | Qdrant Cloud (free tier, 1GB) | For RAG over historical race strategies. Alternatively use ChromaDB locally. |
| **Embeddings** | Google `text-embedding-004` (free via AI Studio) | For embedding historical race strategy documents. |
| **Cache/State** | SQLite + in-memory Python dict | No need for Redis for a personal project. SQLite for persistent historical data cache. |
| **Backend** | FastAPI | REST API + WebSocket for real-time updates. |
| **Agent Framework** | LangGraph (open source) | Handles multi-agent state graphs natively. pip install langgraph. |
| **Frontend** | Streamlit | Simple dashboard + chat interface. No React overhead. |
| **Deployment** | Docker + Render free tier or Railway | Free hosting for demo. |
| **Python Version** | 3.11+ | Use modern Python features (typing, match-case, etc.) |

---

## OpenF1 API Reference

Base URL: `https://api.openf1.org/v1`

### Key Endpoints We Use

```
GET /sessions          — List all sessions (race, qualifying, practice)
    Params: year, country_name, session_type
    
GET /laps              — Lap times for drivers
    Params: session_key, driver_number, lap_number
    
GET /position          — Driver positions throughout session
    Params: session_key, driver_number, date
    
GET /stints            — Tire stint data (compound, start/end laps)
    Params: session_key, driver_number
    
GET /pit               — Pit stop data
    Params: session_key, driver_number
    
GET /intervals         — Gaps between drivers
    Params: session_key, driver_number
    
GET /weather           — Weather conditions during session
    Params: session_key
    
GET /car_data          — Telemetry (speed, throttle, brake, DRS, gear)
    Params: session_key, driver_number, speed
    
GET /drivers           — Driver info (name, team, number)
    Params: session_key
    
GET /meetings          — Race weekend info
    Params: year, country_name
    
GET /race_control      — Flags, safety car, incidents
    Params: session_key
    
GET /championship_drivers  — Championship standings
    Params: session_key
    
GET /championship_teams    — Constructor standings
    Params: session_key
```

### Example Queries

```bash
# Get all 2024 race sessions
curl "https://api.openf1.org/v1/sessions?year=2024&session_type=Race"

# Get Verstappen's laps at 2024 Bahrain GP (session_key from sessions endpoint)
curl "https://api.openf1.org/v1/laps?session_key=9158&driver_number=1"

# Get pit stops for a session
curl "https://api.openf1.org/v1/pit?session_key=9158"

# Get stints (tire compounds used)
curl "https://api.openf1.org/v1/stints?session_key=9158&driver_number=1"

# Get weather during session
curl "https://api.openf1.org/v1/weather?session_key=9158"
```

### Important Notes
- Historical data is free, no auth needed
- Live data (during active sessions) requires paid sponsor tier — we are NOT using live data
- Rate limit: 3 requests/second, 30 requests/minute on free tier
- Implement rate limiting and caching in the client
- Data available from 2023 season onwards

---

## Data Models (Pydantic)

```python
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
from datetime import datetime


class TireCompound(str, Enum):
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"
    UNKNOWN = "UNKNOWN"


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
    lap_time: Optional[float] = None  # in seconds
    sector_times: Optional[SectorTime] = None
    is_pit_in_lap: bool = False
    is_pit_out_lap: bool = False


class DriverState(BaseModel):
    driver_number: int
    name: str
    team: str
    position: int
    gap_to_leader: Optional[float] = None
    gap_to_ahead: Optional[float] = None
    last_lap_time: Optional[float] = None
    tire_compound: TireCompound
    stint_length: int  # laps on current tires
    pit_stops: List[PitStop] = []
    stints: List[Stint] = []
    lap_times: List[LapData] = []
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
    flag: Optional[str] = None  # YELLOW, RED, GREEN, etc.
    category: Optional[str] = None  # SafetyCar, Flag, etc.


class RaceState(BaseModel):
    session_key: int
    meeting_name: str
    track_name: str
    current_lap: int
    total_laps: int
    drivers: List[DriverState] = []
    weather: WeatherState = WeatherState()
    race_control: List[RaceControlMessage] = []
    session_status: str = "Started"  # Started, Finished, Aborted


class TireDegradation(BaseModel):
    driver_number: int
    compound: TireCompound
    deg_rate_per_lap: float  # seconds lost per lap
    predicted_cliff_lap: Optional[int] = None
    laps_remaining_estimate: Optional[int] = None
    current_stint_laps: int


class StrategyRecommendation(BaseModel):
    driver_number: int
    recommended_action: str  # "PIT_NOW", "STAY_OUT", "PIT_IN_X_LAPS"
    recommended_compound: Optional[TireCompound] = None
    optimal_pit_window: Optional[tuple] = None  # (earliest_lap, latest_lap)
    undercut_viable: bool = False
    overcut_viable: bool = False
    reasoning: str
    confidence: float  # 0.0 to 1.0


class HistoricalStrategy(BaseModel):
    """For RAG knowledge base"""
    race_name: str
    year: int
    track: str
    winner: str
    winner_strategy: str  # e.g., "M-H-H" (Medium, Hard, Hard)
    total_laps: int
    pit_stops_winner: int
    weather_conditions: str
    key_events: str  # safety cars, rain, etc.
    summary: str  # natural language summary of winning strategy
```

---

## Agent Architecture

### Overview

6 agents orchestrated via LangGraph state graph:

1. **Router Agent** — Classifies user query type and routes to appropriate agents
2. **Race State Agent** — Fetches and structures current race data from OpenF1
3. **Tire Degradation Agent** — Analyzes lap time trends, predicts tire cliff
4. **Weather Agent** — Monitors weather, detects changes
5. **Strategy Agent (The Brain)** — Synthesizes all data, makes pit recommendations
6. **Explainer Agent** — Converts strategy decision into natural language

### LangGraph State Graph

```
User Query
    |
    v
[Router Agent] --> classifies query type
    |
    |---> "race_status" --> [Race State Agent] --> response
    |---> "tire_analysis" --> [Race State Agent] --> [Tire Deg Agent] --> response
    |---> "weather" --> [Weather Agent] --> response
    |---> "strategy" --> [Race State Agent] + [Tire Deg Agent] + [Weather Agent] + [Strategy RAG Agent]
    |                         |
    |                         v
    |                    [Strategy Agent] --> [Explainer Agent] --> response
    |---> "comparison" --> [Race State Agent] --> [Tire Deg Agent] --> [Explainer Agent] --> response
    |---> "historical" --> [Strategy RAG Agent] --> [Explainer Agent] --> response
```

### Agent Details

#### 1. Router Agent
- Input: User's natural language query + current race context
- Output: Query classification + list of agents to invoke
- System prompt should classify into: race_status, tire_analysis, weather, strategy, comparison, historical
- Simple classification, use Groq Llama for speed

#### 2. Race State Agent
- Responsibility: Fetch and structure live race data from OpenF1
- Tools:
  - `get_sessions(year, country)` — find session_key
  - `get_current_positions(session_key, lap)` — driver positions
  - `get_driver_laps(session_key, driver_number)` — all lap times
  - `get_driver_stints(session_key, driver_number)` — tire compounds used
  - `get_pit_stops(session_key, driver_number)` — pit stop data
  - `get_intervals(session_key)` — gaps between drivers
  - `get_drivers(session_key)` — driver/team info
- Caching: Cache OpenF1 responses in SQLite to avoid re-fetching. Historical data doesn't change.

#### 3. Tire Degradation Agent
- Responsibility: Analyze lap time trends, calculate deg rate, predict tire cliff
- Tools:
  - `calculate_degradation(lap_times, stint_start_lap)` — returns deg rate in seconds/lap
  - `predict_tire_cliff(lap_times, compound, track)` — predicts laps until performance drops
  - `compare_compound_performance(session_key, compound_a, compound_b)` — compare pace across all drivers
- Logic:
  - Filter out pit in/out laps, safety car laps, and first 2 laps of stint (tire warm-up)
  - Use linear regression on clean lap times to get degradation slope
  - Tire cliff = when projected lap time exceeds a threshold (e.g., 2.5s slower than stint best)
  - Compare degradation across drivers on same compound for track-level insights

#### 4. Weather Agent
- Responsibility: Monitor weather, detect rain threats
- Tools:
  - `get_weather_at_lap(session_key, lap)` — conditions at specific lap
  - `detect_weather_change(session_key, from_lap, to_lap)` — flag significant changes
- Logic:
  - Track rainfall boolean, humidity spikes, temperature drops
  - Flag when conditions suggest rain incoming (humidity > 80% + temp dropping)

#### 5. Strategy RAG Agent
- Responsibility: Query historical race strategy knowledge base
- Tools:
  - `query_historical_strategies(track, conditions)` — find similar past races
  - `get_track_characteristics(track)` — typical strategy for this track
- Uses Qdrant vector search over embedded historical race data
- Returns relevant precedents that the Strategy Agent uses for context

#### 6. Strategy Agent (The Brain)
- Responsibility: Synthesize all data and make pit strategy recommendations
- Input: Race state + tire deg data + weather data + historical context
- Tools:
  - `calculate_optimal_pit_window(driver_state, deg_data, race_state)` — when to pit
  - `evaluate_undercut(driver, car_ahead, gap, deg_data)` — is undercut viable?
  - `evaluate_overcut(driver, car_behind, gap, deg_data)` — is overcut viable?
  - `recommend_compound(laps_remaining, weather, track_data)` — what tire to fit
- Logic:
  - Undercut window: gap to car ahead < pit_loss + fresh_tire_advantage (usually ~22s pit loss, ~1-2s/lap fresh tire advantage for 2-3 laps)
  - Optimal pit lap: balance between tire cliff timing and track position
  - Compound choice: laps remaining vs compound expected life vs weather forecast
  - Safety car probability: if many laps remain, factor in SC likelihood (higher at street circuits)

#### 7. Explainer Agent
- Responsibility: Convert strategy data into natural language
- Takes structured StrategyRecommendation and produces human-readable explanation
- Should sound like a real F1 race engineer / commentator
- System prompt: "You are an expert F1 race strategist. Explain your recommendation clearly, referencing specific data points (lap times, gaps, degradation rates). Be concise but thorough. Reference historical precedents when available."

---

## RAG Knowledge Base

### What to Index

1. **Historical race strategies (2023-2025)**
   - Scrape from OpenF1: For each race, build a HistoricalStrategy document
   - Include: winner strategy, pit stop laps, compounds, weather, key events
   - Auto-generate summary using LLM on first indexing

2. **Track characteristics**
   - Manually curated for each circuit on the 2023-2025 calendar
   - Example:
     ```
     Track: Bahrain International Circuit
     Typical strategy: 2-stop (Medium-Hard-Hard or Soft-Hard-Hard)
     Pit loss: ~22 seconds
     Tire degradation: High (rear-limited)
     Overtaking difficulty: Medium (DRS zones help)
     Safety car probability: Medium
     Key factor: Tire management in high track temps
     ```

3. **General F1 strategy principles**
   - Undercut/overcut definitions and when they work
   - Compound characteristics (soft ~15 laps, medium ~25 laps, hard ~35 laps — varies by track)
   - Safety car strategy implications
   - Rain changeover decision framework

### Embedding & Indexing

- Use Google `text-embedding-004` (free) to embed each document
- Store in Qdrant Cloud (free tier, 1GB — more than enough)
- Collection: `f1_strategies`
- Metadata filtering: by track, year, weather conditions

### Indexing Script

Build `scripts/index_historical.py` that:
1. Fetches all race sessions from 2023-2025 via OpenF1
2. For each race, fetches winner's stints, pit stops, weather
3. Constructs a HistoricalStrategy document
4. Generates a natural language summary using Groq
5. Embeds the summary + structured data
6. Upserts into Qdrant

---

## FastAPI Backend

### Endpoints

```
GET  /api/races                    — List all available races (2023-2025)
GET  /api/races/{session_key}      — Get full race state for a specific race
GET  /api/races/{session_key}/lap/{lap_number}  — Get race state at specific lap
POST /api/chat                     — Send chat message, get strategy response
GET  /api/races/{session_key}/summary/{lap_number}  — Auto-generated race summary at lap N
WS   /ws/race/{session_key}        — WebSocket for race replay (sends lap-by-lap updates)
```

### Chat Request/Response

```python
class ChatRequest(BaseModel):
    session_key: int
    current_lap: int
    message: str
    conversation_history: List[dict] = []

class ChatResponse(BaseModel):
    response: str
    strategy_data: Optional[StrategyRecommendation] = None
    agents_used: List[str]  # which agents were invoked
    processing_time_ms: float
```

---

## Streamlit Frontend

### Layout

```
┌──────────────────────────────────────────────────────────┐
│  PITWALL-AI 🏎️        [Select Race ▼]  [Lap Slider: 34] │
├─────────────────────────────┬────────────────────────────┤
│                             │                            │
│  RACE STANDINGS             │  RACE SUMMARY              │
│                             │                            │
│  P1  1  VER  Red Bull       │  "Lap 34/57: Verstappen    │
│      +0.0s   MED  Lap 18   │  leads by 4.2s on 18-lap   │
│  P2  4  NOR  McLaren        │  old mediums. Norris on    │
│      +4.2s   HRD  Lap 6    │  fresh hards, closing at   │
│  P3  16 LEC  Ferrari        │  0.3s per lap. Pit window  │
│      +8.1s   HRD  Lap 5    │  opens in ~3 laps."        │
│  P4  81 PIA  McLaren        │                            │
│      +12.3s  MED  Lap 18   │  WEATHER                   │
│  ...                        │  Air: 28°C  Track: 42°C    │
│                             │  Humidity: 45%  Rain: No   │
│                             │                            │
├─────────────────────────────┴────────────────────────────┤
│                                                          │
│  💬 STRATEGY CHAT                                        │
│                                                          │
│  You: Should Verstappen pit now?                         │
│                                                          │
│  Pitwall: Max's mediums are showing 0.4s/lap deg after   │
│  18 laps. Based on the degradation curve, he has ~3 laps │
│  before the tire cliff. His gap to Norris (4.2s) is      │
│  outside the undercut window (pit loss here is ~22s).    │
│  Recommendation: Pit next lap for hards. At this track   │
│  in 2024, the winning strategy was M-H with a stop on    │
│  lap 20. Confidence: 85%                                 │
│                                                          │
│  You: Compare Norris vs Leclerc tire degradation         │
│                                                          │
│  [Type your question here... ]                    [Send] │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Key Features

1. **Race selector dropdown** — Lists all races 2023-2025
2. **Lap slider** — Scrub through any lap of the race (replay mode)
3. **Auto-updating standings table** — Color-coded by tire compound
4. **Auto-generated summary** — Updates when lap changes, no user action needed
5. **Chat panel** — Natural language strategy questions
6. **Weather widget** — Current conditions

### Streamlit Specifics
- Use `st.columns` for the two-panel layout
- Use `st.chat_message` for the chat interface
- Use `st.slider` for lap selection
- Use `st.selectbox` for race selection
- Use `st.dataframe` for standings with custom styling
- Session state for conversation history

---

## Project Structure

```
pitwall-ai/
├── README.md
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example              # GROQ_API_KEY, GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY
├── pyproject.toml
│
├── src/
│   ├── __init__.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI app, CORS, lifespan
│   │   ├── routes.py         # REST endpoints
│   │   └── websocket.py      # WebSocket for race replay
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── graph.py          # LangGraph state graph definition (main orchestrator)
│   │   ├── state.py          # Shared agent state schema
│   │   ├── router.py         # Router agent — classifies query
│   │   ├── race_state.py     # Race State agent + tools
│   │   ├── tire_deg.py       # Tire Degradation agent + tools
│   │   ├── weather.py        # Weather agent + tools
│   │   ├── strategy_rag.py   # Strategy RAG agent + tools (Qdrant queries)
│   │   ├── strategy.py       # Strategy brain agent + tools
│   │   └── explainer.py      # Explainer agent
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── openf1_client.py  # OpenF1 API wrapper with rate limiting + caching
│   │   ├── models.py         # All Pydantic models (from above)
│   │   ├── cache.py          # SQLite cache layer
│   │   └── race_builder.py   # Builds RaceState from raw OpenF1 data
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── indexer.py        # Index historical data into Qdrant
│   │   ├── embeddings.py     # Embedding generation (Google free)
│   │   ├── retriever.py      # Query historical strategies
│   │   └── knowledge/
│   │       └── tracks.json   # Manually curated track characteristics
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py         # Settings (env vars, API keys)
│   │   ├── llm.py            # LLM client factory (Groq primary, Gemini fallback)
│   │   └── logging.py        # Structured logging setup
│   │
│   └── ui/
│       └── app.py            # Streamlit frontend
│
├── scripts/
│   ├── index_historical.py   # One-time: fetch + index all 2023-2025 races into Qdrant
│   ├── seed_tracks.py        # One-time: seed track characteristics into Qdrant
│   └── test_openf1.py        # Quick script to verify OpenF1 API works
│
└── tests/
    ├── __init__.py
    ├── conftest.py            # Fixtures (mock OpenF1 responses, test race states)
    ├── test_openf1_client.py
    ├── test_race_builder.py
    ├── test_tire_degradation.py
    ├── test_strategy.py
    └── test_agents.py
```

---

## Build Order (Step by Step)

### Phase 1: Data Layer (Week 1)
1. `src/core/config.py` — Environment config with pydantic-settings
2. `src/data/models.py` — All Pydantic models
3. `src/data/openf1_client.py` — OpenF1 API wrapper with:
   - Async HTTP client (httpx)
   - Rate limiting (3 req/s, 30 req/min)
   - SQLite caching (cache responses to avoid re-fetching)
   - All endpoint methods
4. `src/data/cache.py` — SQLite cache layer
5. `src/data/race_builder.py` — Takes raw OpenF1 responses, builds a clean RaceState
6. `scripts/test_openf1.py` — Verify everything works with a real race
7. `tests/test_openf1_client.py` — Unit tests with mocked responses

### Phase 2: Analysis Logic (Week 2)
1. Tire degradation calculation logic (pure Python, no LLM needed):
   - Linear regression on lap times
   - Cliff prediction
   - Compound comparison
2. Weather change detection logic
3. Strategy calculation logic:
   - Undercut/overcut viability calculator
   - Optimal pit window calculator
   - Compound recommendation engine
4. Tests for all analysis functions

### Phase 3: RAG Layer (Week 2-3)
1. `src/core/llm.py` — LLM client with Groq primary, Gemini fallback
2. `src/rag/embeddings.py` — Google text-embedding-004 wrapper
3. `src/rag/indexer.py` — Historical race indexer
4. `src/rag/retriever.py` — Qdrant query wrapper
5. `src/rag/knowledge/tracks.json` — Manual track data
6. `scripts/index_historical.py` — Run once to populate Qdrant
7. `scripts/seed_tracks.py` — Run once to seed track knowledge

### Phase 4: Agents (Week 3-4)
1. `src/agents/state.py` — Shared LangGraph state schema
2. `src/agents/router.py` — Query classifier
3. `src/agents/race_state.py` — Race data fetching agent
4. `src/agents/tire_deg.py` — Tire analysis agent
5. `src/agents/weather.py` — Weather monitoring agent
6. `src/agents/strategy_rag.py` — Historical strategy retrieval agent
7. `src/agents/strategy.py` — The brain
8. `src/agents/explainer.py` — Natural language output
9. `src/agents/graph.py` — Wire everything together in LangGraph
10. `tests/test_agents.py` — Integration tests

### Phase 5: API + UI (Week 4-5)
1. `src/api/main.py` — FastAPI app
2. `src/api/routes.py` — All REST endpoints
3. `src/api/websocket.py` — WebSocket for replay
4. `src/ui/app.py` — Streamlit frontend
5. End-to-end testing

### Phase 6: Polish (Week 5-6)
1. Docker setup (Dockerfile + docker-compose.yml)
2. README.md with:
   - Architecture diagram
   - Screenshots/GIFs of the UI
   - "Quick Start" with docker-compose
   - Technical deep-dive section
3. `.env.example` with all required keys
4. Deploy to Render/Railway free tier
5. Record demo video / GIF for README

---

## Environment Variables

```env
# .env.example
GROQ_API_KEY=gsk_xxxxx
GOOGLE_API_KEY=AIzaxxxxx
QDRANT_URL=https://xxxxx.cloud.qdrant.io
QDRANT_API_KEY=xxxxx

# Optional
LOG_LEVEL=INFO
CACHE_DIR=./data/cache
SQLITE_DB_PATH=./data/cache.db
```

---

## Requirements

```
# requirements.txt
fastapi>=0.115.0
uvicorn>=0.30.0
httpx>=0.27.0
pydantic>=2.0
pydantic-settings>=2.0
langgraph>=0.2.0
langchain-core>=0.3.0
langchain-groq>=0.2.0
langchain-google-genai>=2.0.0
qdrant-client>=1.12.0
streamlit>=1.40.0
numpy>=1.26.0
scipy>=1.14.0
python-dotenv>=1.0.0
aiosqlite>=0.20.0
aiolimiter>=1.1.0
```

---

## Example User Interactions

```
User: "What's the current race situation?"
→ Router: race_status
→ Race State Agent fetches positions, gaps, stints
→ Explainer formats into natural language summary

User: "Should Verstappen pit now?"
→ Router: strategy
→ Race State Agent fetches VER data + field data
→ Tire Deg Agent calculates VER degradation + predicts cliff
→ Weather Agent checks conditions
→ Strategy RAG Agent finds historical precedent for this track
→ Strategy Agent synthesizes and recommends
→ Explainer produces final answer

User: "Compare Norris vs Leclerc tire degradation"
→ Router: comparison
→ Race State Agent fetches NOR + LEC lap times
→ Tire Deg Agent calculates both drivers' deg rates
→ Explainer produces comparison

User: "What happened at this track last year?"
→ Router: historical
→ Strategy RAG Agent queries Qdrant for last year's race
→ Explainer summarizes

User: "What if it rains in 5 laps?"
→ Router: strategy
→ Weather Agent gets current conditions
→ Strategy Agent models rain scenario (switch to intermediates, timing, pit loss)
→ Explainer produces recommendation
```

---

## Key Engineering Decisions to Make During Build

1. **Groq rate limits** — Free tier has token/minute limits. Implement exponential backoff + Gemini fallback. Don't let agent chains fail silently.

2. **OpenF1 caching** — Historical data never changes. Cache aggressively in SQLite. First request hits API, all subsequent reads are local. This also helps with the 30 req/min rate limit.

3. **Tire degradation algorithm** — Start simple (linear regression on clean laps). Can add polynomial fitting later. The hard part is filtering out outliers: pit in/out laps, safety car laps, blue flag laps, first 2 warm-up laps of a stint.

4. **LangGraph state** — Design the shared state carefully. All agents read from and write to this state. Include: race_state, tire_data, weather_data, historical_context, strategy_recommendation, final_response.

5. **Streaming** — Consider streaming the Explainer agent's response to the Streamlit chat for better UX. Groq supports streaming.

6. **Error handling** — OpenF1 can return empty data for some sessions/drivers. Handle gracefully. Some races have red flags, incomplete data, etc.

---

## README Structure (for GitHub)

```markdown
# 🏎️ Pitwall-AI

AI-powered F1 race strategy agent. Ask it "Should Verstappen pit now?" and get data-backed answers.

[Screenshot/GIF here]

## What It Does
[2-3 sentences]

## Architecture
[Diagram showing 6 agents]

## Tech Stack
[Table]

## Quick Start
docker-compose up
Open http://localhost:8501

## How It Works
[Technical deep-dive with code snippets]

## Demo
[Link to deployed version + recorded GIF]

## Data Source
[OpenF1 attribution]
```

---

## Notes for Claude Code

- Start with Phase 1 (data layer). Get OpenF1 working first, cache responses, build a clean RaceState.
- Write tests as you build each component.
- Use type hints everywhere. This project will be code-reviewed by hiring managers.
- Keep functions small and focused. Each agent tool should do one thing.
- Log everything: API calls, cache hits/misses, agent decisions, processing times.
- The analysis logic (tire deg, strategy calc) should be pure Python functions, NOT LLM calls. LLMs are for understanding user queries and explaining results, not for math.
- Use async where it makes sense (OpenF1 client, API routes) but don't over-engineer.
- The Streamlit UI is secondary. Get the backend + agents working perfectly first, then build the UI.

# Pitwall-AI

An AI-powered F1 race strategy agent that answers natural language questions like *"Should Verstappen pit now?"* or *"Compare Norris vs Leclerc tire strategy."*

It connects live OpenF1 race data with a multi-agent LangGraph pipeline, retrieval-augmented historical strategy context, and a Groq-powered LLM to give you pit wall-grade recommendations in seconds.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red)

---

## Demo

```
You:       Should Piastri pit now?

Pitwall:   Given Piastri's current stint on MEDIUM tires (lap 6) and a high Safety
           Car probability at this circuit, I recommend STAY OUT for now. Pitting
           incurs a 30s pit loss and fresh tires won't yield enough time to justify
           it yet. Historical precedent at Sakhir shows a 1-stop Medium-Hard strategy
           wins when the SC window opens around lap 20-25.
           Action: Monitor degradation — reassess on lap 10.

           Agents: router → race_state → tire_degradation → weather → strategy_rag
                   → strategy → explainer  |  6.4s
```

---

## Architecture

```
User question
     │
     ▼
┌──────────┐     classifies query type, extracts driver mentions
│  Router  │
└────┬─────┘
     │
     ▼
┌────────────┐   fetches live lap/stint/pit/weather data from OpenF1
│ Race State │
└────┬───────┘
     │
     ├──────────────────────────────────────┐
     ▼                                      ▼  (strategy/tire queries only)
┌─────────┐                          ┌──────────┐
│ Weather │                          │ Tire Deg │
└────┬────┘                          └────┬─────┘
     │                                    │
     └────────────┬───────────────────────┘
                  │
                  ▼  (strategy/historical queries only)
         ┌──────────────┐   vector search over 2023-2025 race strategies
         │ Strategy RAG │   (Qdrant + Gemini embeddings)
         └──────┬───────┘
                │
                ▼  (strategy/comparison queries only)
         ┌──────────┐   pure-Python pit window math + degradation model
         │ Strategy │
         └──────┬───┘
                │
                ▼
         ┌──────────────┐   Groq Llama 3.3 70B — final natural language answer
         │   Explainer  │
         └──────────────┘
```

**Tech stack**

| Layer | Technology |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | Groq Llama 3.3 70B (primary) · Google Gemini 2.5 Flash (fallback) |
| Embeddings | Google `gemini-embedding-001` (3072-dim) |
| Vector DB | [Qdrant Cloud](https://qdrant.tech) |
| Race data | [OpenF1 API](https://openf1.org) (SQLite-cached) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Container | Docker + Docker Compose |

---

## Features

- **Natural language strategy Q&A** — ask anything, get a pit wall-style answer
- **Live race data** — lap times, tire compounds, gap to leader, pit history via OpenF1
- **Tire degradation modelling** — per-stint deg rate, laps-to-cliff estimate, compound comparison
- **Weather monitoring** — rain threat detection, temperature trend analysis
- **RAG over historical races** — 2023-2025 winning strategies indexed and retrievable by similarity
- **Track characteristics** — 24 circuits with pit loss times, safety car probability, typical strategy
- **Lap replay** — scrub to any lap of any race and ask strategy questions as if you're live
- **Structured recommendations** — every strategy response includes a typed `StrategyRecommendation` with confidence, compound, and pit window
- **SQLite caching** — OpenF1 responses are cached locally so historical queries are instant

---

## Quick Start

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free)
- A [Google AI Studio key](https://aistudio.google.com) (free, for embeddings)
- A [Qdrant Cloud](https://cloud.qdrant.io) free-tier cluster

### 1. Clone and install

```bash
git clone https://github.com/Abhinavs-Repositories/Pitwall.git
cd Pitwall
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your keys:
#   GROQ_API_KEY=gsk_...
#   GOOGLE_API_KEY=AIza...
#   QDRANT_URL=https://xxxxx.cloud.qdrant.io
#   QDRANT_API_KEY=...
```

### 3. Seed the knowledge base (one-time)

```bash
# Load 24 circuit profiles (pit loss, SC probability, typical strategy)
python scripts/seed_tracks.py

# Index 2024 race strategies into Qdrant (takes ~3 min, cached after)
python scripts/index_historical.py --year 2024
```

### 4. Start the servers

```bash
# Terminal 1 — FastAPI backend
uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Streamlit frontend
streamlit run src/ui/app.py
```

Open **http://localhost:8501** in your browser.

---

## Docker

```bash
# Build and start both services
docker-compose up --build

# API at http://localhost:8000
# UI  at http://localhost:8501
```

The Docker setup mounts a named volume for the SQLite cache so race data persists across container restarts.

---

## API Reference

The FastAPI backend exposes a self-documenting interface at **http://localhost:8000/docs**.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/api/races` | List all available races (2023-2025), filterable by `?year=` |
| `GET` | `/api/races/{session_key}` | Full final race state |
| `GET` | `/api/races/{session_key}/lap/{lap}` | Race state at a specific lap (for replay) |
| `GET` | `/api/races/{session_key}/summary/{lap}` | LLM-generated race narrative |
| `POST` | `/api/chat` | Natural language strategy question |

### Chat request / response

```json
POST /api/chat
{
  "session_key": 9472,
  "current_lap": 34,
  "message": "Should Verstappen pit now?",
  "conversation_history": []
}
```

```json
{
  "response": "Verstappen is on lap 16 of HARD tires...",
  "strategy_data": {
    "driver_number": 1,
    "recommended_action": "STAY_OUT",
    "recommended_compound": "HARD",
    "confidence": 0.78,
    "undercut_viable": false,
    "overcut_viable": true,
    "reasoning": "Current tires still have 8+ laps of viable life..."
  },
  "agents_used": ["router", "race_state", "tire_degradation", "weather", "strategy_rag", "strategy", "explainer"],
  "processing_time_ms": 5820.0
}
```

---

## Project Structure

```
pitwall-ai/
├── src/
│   ├── agents/          # LangGraph nodes (router, race_state, tire_deg, weather,
│   │                    #   strategy_rag, strategy, explainer) + graph wiring
│   ├── analysis/        # Pure-Python strategy math, tire degradation, weather analysis
│   ├── api/             # FastAPI app, REST routes, WebSocket endpoint
│   ├── core/            # Config (pydantic-settings), LLM factory, structured logging
│   ├── data/            # OpenF1 client, SQLite cache, Pydantic models, race builder
│   ├── rag/             # Qdrant retriever, embeddings, historical indexer
│   └── ui/              # Streamlit dashboard
├── scripts/
│   ├── seed_tracks.py         # One-time: load 24 circuit profiles into Qdrant
│   └── index_historical.py    # One-time: index race strategies from OpenF1 into Qdrant
├── tests/               # pytest suite (91 tests)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Running Tests

```bash
pytest                    # all tests
pytest tests/test_agents.py -v
pytest -k "test_strategy"
```

The test suite uses `pytest-httpx` to mock OpenF1 API calls — no network access required.

---

## How the Strategy Engine Works

The **Strategy** agent uses pure-Python calculations (no LLM) to produce recommendations:

1. **Tire degradation** — fits a linear model to lap times within the current stint, excluding safety car laps, to estimate deg rate (s/lap) and projected laps until the tire cliff
2. **Pit window** — calculates the lap range where the compound's useful life ends, accounting for track-specific pit loss time
3. **Undercut/overcut viability** — checks gap to the car ahead/behind against the predicted time gain from fresher tires
4. **Weather factor** — flags rain threats that would make an intermediate/wet switch urgent regardless of tire age
5. **Confidence scoring** — combines data quality (laps in stint, gap history) and strategic clarity into a 0–1 confidence score

The **RAG** agent retrieves the 3 most relevant historical races from Qdrant using semantic similarity on the query + track name, then passes them as context to the **Explainer** to ground its answer in real precedent.

---

## Contributing

Pull requests welcome. Key areas for improvement:

- **More historical data** — run `index_historical.py --year 2023` to add a second season
- **Real-time mode** — the OpenF1 API has a live endpoint; wire it to the WebSocket handler in `src/api/websocket.py`
- **Richer degradation model** — replace linear fit with a polynomial or Gaussian process
- **Driver-specific tyre behaviour** — some drivers are historically harder on tyres; this could inform confidence scoring

---

## License

MIT

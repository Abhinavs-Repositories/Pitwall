"""REST API routes for Pitwall-AI.

Endpoints:
    GET  /api/races                               — List all available races (2023-2025)
    GET  /api/races/{session_key}                 — Full race state for a specific race
    GET  /api/races/{session_key}/lap/{lap_number} — Race state at specific lap
    GET  /api/races/{session_key}/summary/{lap_number} — LLM-generated race summary
    POST /api/chat                                — Natural language strategy question
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from src.data.models import ChatRequest, ChatResponse, RaceState, StrategyRecommendation
from src.data.openf1_client import OpenF1Client
from src.data.race_builder import RaceBuilder

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Race listing
# ---------------------------------------------------------------------------


@router.get("/races", response_model=list[dict], tags=["races"])
async def list_races(
    year: int | None = Query(default=None, description="Filter by year (2023–2025)"),
) -> list[dict[str, Any]]:
    """List all available race sessions.

    Returns a flat list of session metadata objects from OpenF1.
    Cached after first fetch — subsequent calls are instant.
    """
    years = [year] if year else [2023, 2024, 2025]
    all_sessions: list[dict] = []

    try:
        async with OpenF1Client() as client:
            for y in years:
                sessions = await client.get_sessions(year=y, session_type="Race")
                all_sessions.extend(sessions)
    except Exception as exc:
        logger.error("Failed to fetch sessions: %s", exc)
        raise HTTPException(status_code=502, detail=f"OpenF1 fetch error: {exc}")

    # Sort by date descending (most recent first) if date field available
    all_sessions.sort(key=lambda s: s.get("date_start", ""), reverse=True)
    return all_sessions


# ---------------------------------------------------------------------------
# Race state
# ---------------------------------------------------------------------------


@router.get("/races/{session_key}", response_model=dict, tags=["races"])
async def get_race_state(session_key: int) -> dict[str, Any]:
    """Return the full final race state for a given session.

    Builds a RaceState object from all OpenF1 data for this session.
    All data is cached — repeated calls are fast.
    """
    try:
        async with OpenF1Client() as client:
            builder = RaceBuilder(client)
            race_state = await builder.build(session_key=session_key)
    except Exception as exc:
        logger.error("Failed to build race state for %s: %s", session_key, exc)
        raise HTTPException(status_code=502, detail=f"Race state build error: {exc}")

    return race_state.model_dump()


@router.get("/races/{session_key}/lap/{lap_number}", response_model=dict, tags=["races"])
async def get_race_state_at_lap(session_key: int, lap_number: int) -> dict[str, Any]:
    """Return race state as it was at the end of a specific lap.

    Used by the Streamlit lap slider to replay any moment of the race.
    """
    if lap_number < 1:
        raise HTTPException(status_code=422, detail="lap_number must be >= 1")

    try:
        async with OpenF1Client() as client:
            builder = RaceBuilder(client)
            race_state = await builder.build(session_key=session_key, up_to_lap=lap_number)
    except Exception as exc:
        logger.error("Failed to build race state at lap %s/%s: %s", lap_number, session_key, exc)
        raise HTTPException(status_code=502, detail=f"Race state error: {exc}")

    return race_state.model_dump()


# ---------------------------------------------------------------------------
# Race summary
# ---------------------------------------------------------------------------


@router.get("/races/{session_key}/summary/{lap_number}", tags=["races"])
async def get_race_summary(request: Request, session_key: int, lap_number: int) -> dict[str, str]:
    """Return an LLM-generated narrative summary of the race state at a given lap.

    Uses the Explainer agent with a fixed "race_status" prompt.
    """
    graph = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent graph not initialised")

    t0 = time.monotonic()
    try:
        result = await graph.ainvoke({
            "session_key": session_key,
            "current_lap": lap_number,
            "user_message": f"Give me a brief narrative summary of the race at lap {lap_number}.",
            "conversation_history": [],
        })
        elapsed = round((time.monotonic() - t0) * 1000, 1)
        return {
            "summary": result.get("final_response", "Summary unavailable"),
            "processing_time_ms": str(elapsed),
        }
    except Exception as exc:
        logger.error("Summary generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Summary error: {exc}")


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Process a natural language strategy question.

    The agent graph is invoked end-to-end:
    Router → Race State → Tire Deg → Weather → Strategy RAG → Strategy → Explainer

    Returns the final natural language response plus optional structured
    StrategyRecommendation data for the primary driver mentioned.
    """
    graph = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        # Lazy compile on first chat request if warmup failed
        try:
            from src.agents.graph import build_graph
            graph = build_graph()
            request.app.state.agent_graph = graph
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Agent graph unavailable: {exc}")

    t0 = time.monotonic()

    try:
        result = await graph.ainvoke({
            "session_key": body.session_key,
            "current_lap": body.current_lap,
            "user_message": body.message,
            "conversation_history": body.conversation_history,
        })
    except Exception as exc:
        logger.error("Chat agent graph error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

    # Extract primary strategy recommendation
    strategy_data: StrategyRecommendation | None = None
    recs = result.get("strategy_recommendations", {})
    if recs:
        strategy_data = next(iter(recs.values()), None)

    return ChatResponse(
        response=result.get("final_response", ""),
        strategy_data=strategy_data,
        agents_used=result.get("agents_used", []),
        processing_time_ms=elapsed_ms,
    )

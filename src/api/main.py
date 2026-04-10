"""FastAPI application entry point.

Includes:
- CORS for Streamlit frontend (localhost:8501)
- Lifespan context manager (warms up the agent graph on startup)
- REST routes + WebSocket router registration
- Structured logging initialisation

Run with:
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as rest_router
from src.api.websocket import router as ws_router
from src.core.config import get_settings
from src.core.logging import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm up shared resources on startup, clean up on shutdown."""
    setup_logging()
    settings = get_settings()
    logger.info(
        "Pitwall-AI API starting",
        extra={"log_level": settings.log_level, "env": "development"},
    )

    # Warm up the agent graph (compiles LangGraph on first import)
    try:
        from src.agents.graph import build_graph
        app.state.agent_graph = build_graph()
        logger.info("Agent graph compiled and ready")
    except Exception as exc:
        logger.warning("Agent graph warm-up failed (will retry on first request): %s", exc)
        app.state.agent_graph = None

    yield

    logger.info("Pitwall-AI API shutting down")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Pitwall-AI",
        description="AI-powered F1 race strategy agent — ask 'Should Verstappen pit now?'",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — allow Streamlit frontend + local dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",  # Streamlit default
            "http://127.0.0.1:8501",
            "http://localhost:3000",  # Future React fallback
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    app.include_router(rest_router, prefix="/api")
    app.include_router(ws_router)

    @app.get("/health", tags=["meta"])
    async def health_check() -> dict:
        return {"status": "ok", "service": "pitwall-ai"}

    return app


# Application instance used by uvicorn
app = create_app()

"""Qdrant-based retriever for historical F1 race strategies.

Usage::

    from src.rag.retriever import StrategyRetriever

    retriever = StrategyRetriever()
    await retriever.init()

    results = await retriever.query("Bahrain 2024 winning strategy")
    track_info = retriever.get_track_characteristics("Bahrain International Circuit")
    await retriever.close()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.core.config import get_settings
from src.data.models import HistoricalStrategy
from src.rag.embeddings import embed_query_sync

logger = logging.getLogger(__name__)

TRACKS_JSON = Path(__file__).parent / "knowledge" / "tracks.json"


class StrategyRetriever:
    """Retrieves relevant historical strategies from Qdrant.

    Falls back gracefully if Qdrant is unavailable — returns empty results
    instead of crashing the agent graph.
    """

    def __init__(self) -> None:
        self._client: Any = None  # qdrant_client.QdrantClient
        self._tracks: dict[str, dict] = {}
        settings = get_settings()
        self._collection = settings.qdrant_collection
        self._qdrant_url = settings.qdrant_url
        self._qdrant_api_key = settings.qdrant_api_key

    async def init(self) -> None:
        """Connect to Qdrant and load track characteristics."""
        self._load_tracks()
        if self._qdrant_url:
            try:
                from qdrant_client import AsyncQdrantClient

                self._client = AsyncQdrantClient(
                    url=self._qdrant_url,
                    api_key=self._qdrant_api_key or None,
                )
                logger.info("Qdrant retriever connected: %s", self._qdrant_url)
            except Exception as exc:
                logger.warning("Qdrant unavailable — RAG will be skipped: %s", exc)
        else:
            logger.warning("QDRANT_URL not configured — historical RAG disabled")

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def query_historical_strategies(
        self,
        query: str,
        track: str | None = None,
        year: int | None = None,
        top_k: int = 3,
    ) -> list[HistoricalStrategy]:
        """Retrieve the most relevant historical race strategies.

        Args:
            query:  Natural language query (e.g., "2-stop strategy Bahrain").
            track:  Optional filter by track name.
            year:   Optional filter by year.
            top_k:  Number of results to return.

        Returns:
            List of HistoricalStrategy objects, most relevant first.
        """
        if not self._client:
            logger.debug("Qdrant not available — returning empty strategy list")
            return []

        try:
            query_vector = embed_query_sync(query)

            response = await self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )

            strategies = []
            for hit in response.points:
                payload = hit.payload or {}
                try:
                    strategies.append(HistoricalStrategy(**payload))
                except Exception as exc:
                    logger.warning("Could not parse strategy payload: %s", exc)

            logger.info("RAG query returned %d strategies for %r", len(strategies), query[:60])
            return strategies

        except Exception as exc:
            logger.error("Qdrant search failed: %s", exc)
            return []

    def get_track_characteristics(self, track_name: str) -> dict | None:
        """Return manually curated track data for a given circuit.

        Matches on exact name or case-insensitive partial match.
        """
        # Exact match first
        if track_name in self._tracks:
            return self._tracks[track_name]

        # Case-insensitive partial match
        lower = track_name.lower()
        for name, data in self._tracks.items():
            if lower in name.lower() or name.lower() in lower:
                return data

        # Match on country
        for name, data in self._tracks.items():
            if lower in data.get("country", "").lower():
                return data

        logger.debug("No track characteristics found for: %s", track_name)
        return None

    def list_tracks(self) -> list[str]:
        """Return all track names in the knowledge base."""
        return list(self._tracks.keys())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_tracks(self) -> None:
        """Load tracks.json into memory."""
        try:
            data = json.loads(TRACKS_JSON.read_text(encoding="utf-8"))
            self._tracks = {t["name"]: t for t in data.get("tracks", [])}
            logger.debug("Loaded %d track profiles", len(self._tracks))
        except Exception as exc:
            logger.error("Failed to load tracks.json: %s", exc)
            self._tracks = {}



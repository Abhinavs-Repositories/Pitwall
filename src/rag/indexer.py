"""Historical race strategy indexer — populates the Qdrant vector store.

Run via: python scripts/index_historical.py

Flow for each race session:
  1. Fetch winner's stints + pit stops from OpenF1.
  2. Fetch weather summary.
  3. Fetch race control messages (SC, red flags).
  4. Build a HistoricalStrategy document.
  5. Generate a natural language summary via LLM.
  6. Embed the summary + structured fields.
  7. Upsert into Qdrant collection `f1_strategies`.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from src.core.config import get_settings
from src.core.llm import get_llm
from src.data.models import HistoricalStrategy, TireCompound
from src.rag.embeddings import embed_texts_sync, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class StrategyIndexer:
    """Indexes HistoricalStrategy documents into Qdrant."""

    def __init__(self) -> None:
        self._client: Any = None
        settings = get_settings()
        self._collection = settings.qdrant_collection
        self._qdrant_url = settings.qdrant_url
        self._qdrant_api_key = settings.qdrant_api_key

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Connect to Qdrant and create collection if it doesn't exist."""
        if not self._qdrant_url:
            raise RuntimeError("QDRANT_URL is not set in .env")

        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._client = AsyncQdrantClient(
                url=self._qdrant_url,
                api_key=self._qdrant_api_key or None,
            )

            # Create collection if needed
            collections = await self._client.get_collections()
            existing = [c.name for c in collections.collections]
            if self._collection not in existing:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection: %s", self._collection)
            else:
                logger.info("Using existing Qdrant collection: %s", self._collection)

        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required. Run: pip install qdrant-client"
            ) from exc

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_strategy(self, strategy: HistoricalStrategy) -> str:
        """Embed and upsert a single HistoricalStrategy.

        Returns the Qdrant point ID (UUID string).
        """
        self._ensure_open()
        from qdrant_client.models import PointStruct

        text = _strategy_to_text(strategy)
        vectors = embed_texts_sync([text])
        vector = vectors[0]

        point_id = str(uuid.uuid4())
        payload = strategy.model_dump()

        await self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

        logger.info(
            "Indexed strategy",
            extra={"race": strategy.race_name, "year": strategy.year, "id": point_id},
        )
        return point_id

    async def index_many(self, strategies: list[HistoricalStrategy]) -> list[str]:
        """Batch-index a list of strategies.

        Returns list of Qdrant point IDs.
        """
        self._ensure_open()
        from qdrant_client.models import PointStruct

        texts = [_strategy_to_text(s) for s in strategies]
        vectors = embed_texts_sync(texts)

        points = []
        ids = []
        for strategy, vector in zip(strategies, vectors):
            point_id = str(uuid.uuid4())
            ids.append(point_id)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=strategy.model_dump(),
                )
            )

        await self._client.upsert(
            collection_name=self._collection,
            points=points,
        )

        logger.info("Batch-indexed %d strategies", len(strategies))
        return ids

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._client is None:
            raise RuntimeError("StrategyIndexer.init() must be called first")


# ---------------------------------------------------------------------------
# LLM-based summary generation
# ---------------------------------------------------------------------------


def generate_strategy_summary(strategy: HistoricalStrategy) -> str:
    """Use Groq to generate a natural language summary for a race strategy.

    Called once per race during indexing. The summary is stored in the
    HistoricalStrategy.summary field and embedded into Qdrant.
    """
    llm = get_llm()
    prompt = (
        f"You are an F1 strategy expert. Write a 3-4 sentence natural language summary "
        f"of the following race strategy for the purpose of a strategy knowledge base.\n\n"
        f"Race: {strategy.race_name} {strategy.year}\n"
        f"Track: {strategy.track}\n"
        f"Winner: {strategy.winner}\n"
        f"Strategy: {strategy.winner_strategy}\n"
        f"Total laps: {strategy.total_laps}\n"
        f"Pit stops (winner): {strategy.pit_stops_winner}\n"
        f"Weather: {strategy.weather_conditions}\n"
        f"Key events: {strategy.key_events}\n\n"
        f"Focus on: tire compounds used, pit stop timing rationale, "
        f"weather impact, and what made this strategy successful."
    )

    response = llm.invoke(prompt)
    return str(response.content).strip()


# ---------------------------------------------------------------------------
# Text representation for embedding
# ---------------------------------------------------------------------------


def _strategy_to_text(strategy: HistoricalStrategy) -> str:
    """Convert a HistoricalStrategy to a single embeddable text string."""
    return (
        f"Race: {strategy.race_name} {strategy.year}. "
        f"Track: {strategy.track}. "
        f"Winner: {strategy.winner}. "
        f"Strategy: {strategy.winner_strategy}. "
        f"Pit stops: {strategy.pit_stops_winner}. "
        f"Total laps: {strategy.total_laps}. "
        f"Weather: {strategy.weather_conditions}. "
        f"Key events: {strategy.key_events}. "
        f"{strategy.summary}"
    )


# ---------------------------------------------------------------------------
# Helper: build HistoricalStrategy from OpenF1 raw data
# ---------------------------------------------------------------------------


def build_historical_strategy(
    session: dict,
    winner_driver: dict,
    winner_stints: list[dict],
    winner_pits: list[dict],
    weather_summary: str,
    key_events: str,
    summary: str = "",
) -> HistoricalStrategy:
    """Construct a HistoricalStrategy from raw OpenF1 response dicts."""
    compounds = [s.get("compound", "?").upper()[0] for s in winner_stints if s.get("compound")]
    strategy_str = "-".join(compounds) if compounds else "Unknown"

    return HistoricalStrategy(
        race_name=session.get("meeting_name", "Unknown Race"),
        year=session.get("year", 0),
        track=session.get("location", "Unknown Track"),
        winner=winner_driver.get("full_name", "Unknown"),
        winner_strategy=strategy_str,
        total_laps=session.get("laps", 0),
        pit_stops_winner=len(winner_pits),
        weather_conditions=weather_summary,
        key_events=key_events,
        summary=summary,
    )

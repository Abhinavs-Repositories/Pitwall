"""Google text-embedding-004 wrapper for generating vector embeddings.

Free via Google AI Studio (GOOGLE_API_KEY). Dimension: 768.

Usage::

    from src.rag.embeddings import embed_texts, embed_query

    vectors = await embed_texts(["Strategy: M-H-H at Bahrain 2024"])
    query_vec = await embed_query("Bahrain winning strategy")
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.core.config import get_settings

logger = logging.getLogger(__name__)

# Google text-embedding-004 produces 768-dimensional vectors
EMBEDDING_DIM = 768


def _get_genai():
    """Lazy import google.generativeai to avoid hard startup dependency."""
    try:
        import google.generativeai as genai
        return genai
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is required for embeddings. "
            "Run: pip install google-generativeai"
        ) from exc


def _configure_genai() -> Any:
    settings = get_settings()
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set — embeddings unavailable")
    genai = _get_genai()
    genai.configure(api_key=settings.google_api_key)
    return genai


def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    """Synchronously embed a list of texts.

    Args:
        texts: List of strings to embed. Each string ≤ 2048 tokens.

    Returns:
        List of 768-dimensional float vectors, one per input text.
    """
    genai = _configure_genai()
    settings = get_settings()
    model = settings.embedding_model

    results: list[list[float]] = []
    # Batch in groups of 100 (API limit)
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = genai.embed_content(
                model=model,
                content=batch,
                task_type="RETRIEVAL_DOCUMENT",
            )
            results.extend(response["embedding"] if isinstance(batch, list) else [response["embedding"]])
        except Exception as exc:
            logger.error("Embedding failed for batch %d: %s", i // batch_size, exc)
            raise

    logger.debug("Embedded %d texts", len(texts))
    return results


def embed_query_sync(query: str) -> list[float]:
    """Synchronously embed a single query string for retrieval.

    Uses RETRIEVAL_QUERY task type (slightly different representation than DOCUMENT).
    """
    genai = _configure_genai()
    settings = get_settings()

    response = genai.embed_content(
        model=settings.embedding_model,
        content=query,
        task_type="RETRIEVAL_QUERY",
    )
    return response["embedding"]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Async wrapper around embed_texts_sync."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_texts_sync, texts)


async def embed_query(query: str) -> list[float]:
    """Async wrapper around embed_query_sync."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_query_sync, query)

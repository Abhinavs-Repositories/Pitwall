"""Google text-embedding-004 wrapper for generating vector embeddings.

Free via Google AI Studio (GOOGLE_API_KEY). Dimension: 768.
Uses the new google-genai SDK (google.genai).

Usage::

    from src.rag.embeddings import embed_texts, embed_query

    vectors = await embed_texts(["Strategy: M-H-H at Bahrain 2024"])
    query_vec = await embed_query("Bahrain winning strategy")
"""

from __future__ import annotations

import asyncio
import logging

from src.core.config import get_settings

logger = logging.getLogger(__name__)

# gemini-embedding-001 produces 3072-dimensional vectors
EMBEDDING_DIM = 3072


def _get_client():
    """Build a google.genai Client (new SDK)."""
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "google-genai is required for embeddings. "
            "Run: pip install google-genai"
        ) from exc

    settings = get_settings()
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set — embeddings unavailable")

    return genai.Client(api_key=settings.google_api_key)


def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    """Synchronously embed a list of texts.

    Args:
        texts: List of strings to embed. Each string ≤ 2048 tokens.

    Returns:
        List of 768-dimensional float vectors, one per input text.
    """
    client = _get_client()
    settings = get_settings()
    model = settings.embedding_model

    results: list[list[float]] = []
    # Embed one at a time (new SDK doesn't batch via list in embed_content)
    for i, text in enumerate(texts):
        try:
            response = client.models.embed_content(
                model=model,
                contents=text,
            )
            # New SDK returns EmbedContentResponse with .embeddings list
            embedding = response.embeddings[0].values
            results.append(list(embedding))
        except Exception as exc:
            logger.error("Embedding failed for text %d: %s", i, exc)
            raise

    logger.debug("Embedded %d texts", len(texts))
    return results


def embed_query_sync(query: str) -> list[float]:
    """Synchronously embed a single query string for retrieval."""
    client = _get_client()
    settings = get_settings()

    response = client.models.embed_content(
        model=settings.embedding_model,
        contents=query,
    )
    return list(response.embeddings[0].values)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Async wrapper around embed_texts_sync."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_texts_sync, texts)


async def embed_query(query: str) -> list[float]:
    """Async wrapper around embed_query_sync."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_query_sync, query)

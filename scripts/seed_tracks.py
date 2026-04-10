#!/usr/bin/env python
"""One-time script: seed track characteristics from tracks.json into Qdrant.

Each track profile is embedded and stored as a separate document with
payload type="track_characteristics" so the retriever can distinguish
them from race strategy documents.

Usage:
    python scripts/seed_tracks.py
    python scripts/seed_tracks.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import get_settings
from src.core.logging import setup_logging
from src.rag.embeddings import embed_texts_sync, EMBEDDING_DIM

logger = logging.getLogger(__name__)

TRACKS_JSON = Path(__file__).parent.parent / "src" / "rag" / "knowledge" / "tracks.json"


async def run(dry_run: bool) -> None:
    setup_logging()
    settings = get_settings()

    data = json.loads(TRACKS_JSON.read_text(encoding="utf-8"))
    tracks = data.get("tracks", [])
    logger.info("Loaded %d track profiles from tracks.json", len(tracks))

    if dry_run:
        for track in tracks:
            print(f"\n{track['name']} ({track['country']})")
            print(f"  Strategy: {track['typical_strategy']}")
            print(f"  Pit loss: {track['pit_loss_seconds']}s")
            print(f"  SC prob: {track['safety_car_probability']}")
            print(f"  Key factor: {track['key_factor']}")
        return

    if not settings.qdrant_url:
        logger.error("QDRANT_URL not set — cannot seed tracks")
        sys.exit(1)

    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except ImportError:
        logger.error("qdrant-client not installed. Run: pip install qdrant-client")
        sys.exit(1)

    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )

    # Ensure collection exists
    collections = await client.get_collections()
    existing = [c.name for c in collections.collections]
    if settings.qdrant_collection not in existing:
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("Created collection: %s", settings.qdrant_collection)

    # Build text representations and embed
    texts = [_track_to_text(t) for t in tracks]
    logger.info("Embedding %d track profiles ...", len(tracks))
    vectors = embed_texts_sync(texts)

    points = []
    for track, vector in zip(tracks, vectors):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={**track, "doc_type": "track_characteristics"},
            )
        )

    await client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info("Seeded %d track profiles into Qdrant", len(points))
    await client.close()


def _track_to_text(track: dict) -> str:
    return (
        f"Track: {track['name']} in {track['country']}. "
        f"Typical strategy: {track['typical_strategy']}. "
        f"Pit loss: {track['pit_loss_seconds']} seconds. "
        f"Tire degradation: {track['tire_degradation']}. "
        f"Overtaking difficulty: {track['overtaking_difficulty']}. "
        f"Safety car probability: {track['safety_car_probability']}. "
        f"Key factor: {track['key_factor']}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed track characteristics into Qdrant")
    parser.add_argument("--dry-run", action="store_true", help="Print without seeding")
    args = parser.parse_args()
    asyncio.run(run(dry_run=args.dry_run))


if __name__ == "__main__":
    main()

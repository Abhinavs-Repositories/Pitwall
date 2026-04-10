"""SQLite-backed async cache for OpenF1 API responses.

Historical F1 data never changes, so we cache aggressively:
- Key   = (endpoint, frozenset of query params)
- Value = JSON-serialised response body
- TTL   = None (historical data is immutable; live data not supported)
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import aiosqlite

from src.core.config import get_settings

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS api_cache (
    cache_key   TEXT PRIMARY KEY,
    endpoint    TEXT NOT NULL,
    params_hash TEXT NOT NULL,
    response    TEXT NOT NULL,
    created_at  REAL NOT NULL DEFAULT (unixepoch('now'))
);
"""

_GET = "SELECT response FROM api_cache WHERE cache_key = ?;"
_SET = """
INSERT OR REPLACE INTO api_cache (cache_key, endpoint, params_hash, response)
VALUES (?, ?, ?, ?);
"""


def _make_key(endpoint: str, params: dict[str, Any]) -> tuple[str, str]:
    """Return (cache_key, params_hash) for the given endpoint + params."""
    # Sort params so key is deterministic regardless of dict ordering
    serialised = json.dumps(params, sort_keys=True)
    params_hash = hashlib.sha256(serialised.encode()).hexdigest()
    cache_key = f"{endpoint}:{params_hash}"
    return cache_key, params_hash


class OpenF1Cache:
    """Async SQLite cache.  Call ``await cache.init()`` before first use."""

    def __init__(self, db_path: str | None = None) -> None:
        settings = get_settings()
        self._db_path = db_path or settings.sqlite_db_path
        self._conn: aiosqlite.Connection | None = None

    async def init(self) -> None:
        """Open the database and create tables if required."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        await self._conn.execute(_CREATE_TABLE)
        await self._conn.commit()
        logger.info("Cache initialised", extra={"db_path": self._db_path})

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def get(self, endpoint: str, params: dict[str, Any]) -> Any | None:
        """Return cached response data, or None on cache miss."""
        self._ensure_open()
        cache_key, _ = _make_key(endpoint, params)
        async with self._conn.execute(_GET, (cache_key,)) as cursor:  # type: ignore[union-attr]
            row = await cursor.fetchone()
        if row:
            logger.debug("Cache hit", extra={"endpoint": endpoint, "params": params})
            return json.loads(row[0])
        logger.debug("Cache miss", extra={"endpoint": endpoint, "params": params})
        return None

    async def set(self, endpoint: str, params: dict[str, Any], data: Any) -> None:
        """Persist a response to the cache."""
        self._ensure_open()
        cache_key, params_hash = _make_key(endpoint, params)
        await self._conn.execute(  # type: ignore[union-attr]
            _SET, (cache_key, endpoint, params_hash, json.dumps(data))
        )
        await self._conn.commit()  # type: ignore[union-attr]
        logger.debug("Cache stored", extra={"endpoint": endpoint, "cache_key": cache_key})

    async def clear_endpoint(self, endpoint: str) -> int:
        """Delete all cached entries for a given endpoint. Returns rows deleted."""
        self._ensure_open()
        cursor = await self._conn.execute(  # type: ignore[union-attr]
            "DELETE FROM api_cache WHERE endpoint = ?;", (endpoint,)
        )
        await self._conn.commit()  # type: ignore[union-attr]
        count = cursor.rowcount
        logger.info("Cache cleared", extra={"endpoint": endpoint, "rows": count})
        return count

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "OpenF1Cache":
        await self.init()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._conn is None:
            raise RuntimeError("Cache not initialised — call await cache.init() first.")

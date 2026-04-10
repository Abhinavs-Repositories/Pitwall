"""OpenF1 API async client with rate limiting and SQLite caching.

Rate limits (free tier):
  - 3 requests / second
  - 30 requests / minute

All historical data is cached in SQLite on first fetch and served locally
on subsequent calls, keeping us well within rate limits.
"""

import asyncio
import logging
import time
from typing import Any

import httpx
from aiolimiter import AsyncLimiter

from src.core.config import get_settings
from src.data.cache import OpenF1Cache

logger = logging.getLogger(__name__)

# Type alias for raw API responses (list of dicts)
RawList = list[dict[str, Any]]


class OpenF1Client:
    """Async HTTP client for the OpenF1 API.

    Usage::

        async with OpenF1Client() as client:
            sessions = await client.get_sessions(year=2024, session_type="Race")
    """

    def __init__(self, cache: OpenF1Cache | None = None) -> None:
        settings = get_settings()
        self._base_url = settings.openf1_base_url.rstrip("/")
        # Token bucket: max 3 requests per second
        self._per_second_limiter = AsyncLimiter(
            max_rate=settings.openf1_rate_limit_per_second, time_period=1
        )
        # Sliding window: max 30 requests per minute
        self._per_minute_limiter = AsyncLimiter(
            max_rate=settings.openf1_rate_limit_per_minute, time_period=60
        )
        self._cache = cache or OpenF1Cache()
        self._http: httpx.AsyncClient | None = None
        self._owns_cache = cache is None  # close cache on exit if we created it

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "OpenF1Client":
        await self._cache.init()
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(30.0),
            headers={"Accept": "application/json"},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._http:
            await self._http.aclose()
        if self._owns_cache:
            await self._cache.close()

    # ------------------------------------------------------------------
    # High-level endpoint methods
    # ------------------------------------------------------------------

    async def get_sessions(
        self,
        year: int | None = None,
        country_name: str | None = None,
        session_type: str | None = None,
        session_key: int | None = None,
    ) -> RawList:
        """List sessions. Filter by year, country, type, or specific key."""
        params: dict[str, Any] = {}
        if year is not None:
            params["year"] = year
        if country_name is not None:
            params["country_name"] = country_name
        if session_type is not None:
            params["session_type"] = session_type
        if session_key is not None:
            params["session_key"] = session_key
        return await self._get("/sessions", params)

    async def get_laps(
        self,
        session_key: int,
        driver_number: int | None = None,
        lap_number: int | None = None,
    ) -> RawList:
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        if lap_number is not None:
            params["lap_number"] = lap_number
        return await self._get("/laps", params)

    async def get_position(
        self,
        session_key: int,
        driver_number: int | None = None,
    ) -> RawList:
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return await self._get("/position", params)

    async def get_stints(
        self,
        session_key: int,
        driver_number: int | None = None,
    ) -> RawList:
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return await self._get("/stints", params)

    async def get_pit(
        self,
        session_key: int,
        driver_number: int | None = None,
    ) -> RawList:
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return await self._get("/pit", params)

    async def get_intervals(
        self,
        session_key: int,
        driver_number: int | None = None,
    ) -> RawList:
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return await self._get("/intervals", params)

    async def get_weather(self, session_key: int) -> RawList:
        return await self._get("/weather", {"session_key": session_key})

    async def get_car_data(
        self,
        session_key: int,
        driver_number: int | None = None,
    ) -> RawList:
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return await self._get("/car_data", params)

    async def get_drivers(self, session_key: int) -> RawList:
        return await self._get("/drivers", {"session_key": session_key})

    async def get_meetings(
        self,
        year: int | None = None,
        country_name: str | None = None,
    ) -> RawList:
        params: dict[str, Any] = {}
        if year is not None:
            params["year"] = year
        if country_name is not None:
            params["country_name"] = country_name
        return await self._get("/meetings", params)

    async def get_race_control(self, session_key: int) -> RawList:
        return await self._get("/race_control", {"session_key": session_key})

    async def get_championship_drivers(self, session_key: int) -> RawList:
        return await self._get("/championship_drivers", {"session_key": session_key})

    async def get_championship_teams(self, session_key: int) -> RawList:
        return await self._get("/championship_teams", {"session_key": session_key})

    # ------------------------------------------------------------------
    # Internal HTTP layer
    # ------------------------------------------------------------------

    async def _get(self, endpoint: str, params: dict[str, Any]) -> RawList:
        """Fetch from cache or OpenF1 API, applying rate limiting."""
        self._ensure_open()

        # Try cache first
        cached = await self._cache.get(endpoint, params)
        if cached is not None:
            return cached

        # Rate limiting before every real network request
        async with self._per_second_limiter:
            async with self._per_minute_limiter:
                data = await self._fetch(endpoint, params)

        # Cache the result
        await self._cache.set(endpoint, params, data)
        return data

    async def _fetch(
        self,
        endpoint: str,
        params: dict[str, Any],
        *,
        retries: int = 3,
        backoff_base: float = 2.0,
    ) -> RawList:
        """Execute the HTTP request with exponential backoff on transient errors."""
        url = endpoint  # httpx uses base_url so this is a relative path
        last_exc: Exception | None = None

        for attempt in range(retries):
            try:
                t0 = time.monotonic()
                response = await self._http.get(url, params=params)  # type: ignore[union-attr]
                elapsed_ms = (time.monotonic() - t0) * 1000

                if response.status_code == 200:
                    data: RawList = response.json()
                    logger.info(
                        "OpenF1 API request",
                        extra={
                            "endpoint": endpoint,
                            "params": params,
                            "status": 200,
                            "rows": len(data),
                            "elapsed_ms": round(elapsed_ms, 1),
                        },
                    )
                    return data

                if response.status_code == 404:
                    # OpenF1 returns 404 when no data exists for the query params
                    logger.debug("OpenF1 returned 404 (no data) for %s %s", endpoint, params)
                    return []

                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", backoff_base ** attempt))
                    logger.warning(
                        "OpenF1 rate limited, backing off",
                        extra={"retry_after": retry_after, "attempt": attempt},
                    )
                    await asyncio.sleep(retry_after)
                    continue

                # Other 4xx/5xx — raise immediately
                response.raise_for_status()

            except httpx.HTTPStatusError:
                raise
            except httpx.TransportError as exc:
                last_exc = exc
                wait = backoff_base ** attempt
                logger.warning(
                    "OpenF1 transport error, retrying",
                    extra={"error": str(exc), "attempt": attempt, "wait_s": wait},
                )
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"OpenF1 request failed after {retries} attempts: {endpoint} {params}"
        ) from last_exc

    def _ensure_open(self) -> None:
        if self._http is None:
            raise RuntimeError(
                "OpenF1Client not initialised — use 'async with OpenF1Client() as client'."
            )

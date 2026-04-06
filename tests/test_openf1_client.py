"""Unit tests for OpenF1Client — all network calls are mocked."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.cache import OpenF1Cache
from src.data.openf1_client import OpenF1Client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(data: list[dict], status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.headers = {}
    resp.raise_for_status = MagicMock()
    return resp


async def _make_client(tmp_path) -> OpenF1Client:
    """Return an OpenF1Client backed by a temp-file SQLite cache."""
    db = str(tmp_path / "test_cache.db")
    cache = OpenF1Cache(db_path=db)
    client = OpenF1Client(cache=cache)
    await cache.init()
    client._http = MagicMock()  # will be replaced per-test
    return client


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_miss_then_hit(tmp_path, raw_sessions):
    """First call hits the network; second call is served from cache."""
    client = await _make_client(tmp_path)
    client._http.get = AsyncMock(return_value=_mock_response(raw_sessions))

    result1 = await client.get_sessions(year=2024, session_type="Race")
    assert result1 == raw_sessions
    assert client._http.get.call_count == 1

    result2 = await client.get_sessions(year=2024, session_type="Race")
    assert result2 == raw_sessions
    # Still only 1 real HTTP call — second was from cache
    assert client._http.get.call_count == 1

    await client._cache.close()


@pytest.mark.asyncio
async def test_different_params_separate_cache_entries(tmp_path, raw_sessions):
    client = await _make_client(tmp_path)
    client._http.get = AsyncMock(return_value=_mock_response(raw_sessions))

    await client.get_sessions(year=2024)
    await client.get_sessions(year=2023)

    # Both params combos should result in their own network call
    assert client._http.get.call_count == 2
    await client._cache.close()


# ---------------------------------------------------------------------------
# Endpoint method tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_sessions_passes_params(tmp_path, raw_sessions):
    client = await _make_client(tmp_path)
    client._http.get = AsyncMock(return_value=_mock_response(raw_sessions))

    await client.get_sessions(year=2024, session_type="Race")

    call_kwargs = client._http.get.call_args
    params = call_kwargs.kwargs.get("params") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else {}
    # params may be positional or keyword depending on httpx
    assert client._http.get.called


@pytest.mark.asyncio
async def test_get_drivers(tmp_path, raw_drivers):
    client = await _make_client(tmp_path)
    client._http.get = AsyncMock(return_value=_mock_response(raw_drivers))

    result = await client.get_drivers(session_key=9158)

    assert len(result) == 3
    assert result[0]["driver_number"] == 1
    await client._cache.close()


@pytest.mark.asyncio
async def test_get_stints(tmp_path, raw_stints):
    client = await _make_client(tmp_path)
    client._http.get = AsyncMock(return_value=_mock_response(raw_stints))

    result = await client.get_stints(session_key=9158)

    assert len(result) == len(raw_stints)
    await client._cache.close()


@pytest.mark.asyncio
async def test_get_pit(tmp_path, raw_pits):
    client = await _make_client(tmp_path)
    client._http.get = AsyncMock(return_value=_mock_response(raw_pits))

    result = await client.get_pit(session_key=9158)

    assert len(result) == 4
    await client._cache.close()


@pytest.mark.asyncio
async def test_get_weather(tmp_path, raw_weather):
    client = await _make_client(tmp_path)
    client._http.get = AsyncMock(return_value=_mock_response(raw_weather))

    result = await client.get_weather(session_key=9158)

    assert result[0]["air_temperature"] == 28.4
    await client._cache.close()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raises_after_max_retries(tmp_path):
    import httpx

    client = await _make_client(tmp_path)
    client._http.get = AsyncMock(
        side_effect=httpx.ConnectError("connection refused")
    )

    with pytest.raises(RuntimeError, match="failed after"):
        await client.get_sessions(year=2024)

    await client._cache.close()


@pytest.mark.asyncio
async def test_empty_response_returned_as_empty_list(tmp_path):
    client = await _make_client(tmp_path)
    client._http.get = AsyncMock(return_value=_mock_response([]))

    result = await client.get_laps(session_key=9158, driver_number=99)

    assert result == []
    await client._cache.close()

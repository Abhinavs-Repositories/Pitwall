#!/usr/bin/env python
"""One-time script: fetch all 2023–2025 race strategies from OpenF1 and index into Qdrant.

Usage:
    python scripts/index_historical.py
    python scripts/index_historical.py --year 2024
    python scripts/index_historical.py --dry-run   # print strategies without indexing
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import get_settings
from src.core.logging import configure_logging as setup_logging
from src.data.cache import OpenF1Cache
from src.data.openf1_client import OpenF1Client
from src.rag.indexer import StrategyIndexer, build_historical_strategy, generate_strategy_summary

logger = logging.getLogger(__name__)


async def run(years: list[int], dry_run: bool) -> None:
    setup_logging()
    settings = get_settings()

    indexer = StrategyIndexer()
    if not dry_run:
        await indexer.init()

    async with OpenF1Client() as client:
        for year in years:
            logger.info("Processing year %d ...", year)
            sessions = await client.get_sessions(year=year, session_type="Race")

            if not sessions:
                logger.warning("No race sessions found for %d", year)
                continue

            logger.info("Found %d races in %d", len(sessions), year)

            for session in sessions:
                session_key = session.get("session_key")
                meeting_name = session.get("meeting_name", "?")

                if not session_key:
                    continue

                logger.info("  Processing: %s (session_key=%s)", meeting_name, session_key)

                try:
                    strategy = await _process_session(client, session)
                except Exception as exc:
                    logger.error("  Failed %s: %s", meeting_name, exc)
                    continue

                if dry_run:
                    print(f"\n{'='*60}")
                    print(f"Race: {strategy.race_name} {strategy.year}")
                    print(f"Winner: {strategy.winner}")
                    print(f"Strategy: {strategy.winner_strategy}")
                    print(f"Weather: {strategy.weather_conditions}")
                    print(f"Events: {strategy.key_events}")
                    print(f"Summary: {strategy.summary[:200]}...")
                else:
                    await indexer.index_strategy(strategy)
                    logger.info("  Indexed: %s", meeting_name)

    if not dry_run:
        await indexer.close()

    logger.info("Done.")


async def _process_session(client: OpenF1Client, session: dict) -> object:
    """Fetch data for one session and build a HistoricalStrategy."""
    from src.rag.indexer import build_historical_strategy, generate_strategy_summary

    session_key = session["session_key"]

    # Get finishing positions to identify winner
    positions = await client.get_position(session_key=session_key)

    # Find the driver who finished P1 (last position entry per driver)
    driver_positions: dict[int, int] = {}
    for pos in positions:
        dn = pos.get("driver_number")
        p = pos.get("position")
        if dn and p:
            driver_positions[dn] = p

    if not driver_positions:
        raise ValueError(f"No position data for session {session_key}")

    winner_number = min(driver_positions, key=driver_positions.get)

    # Get driver info
    drivers = await client.get_drivers(session_key=session_key)
    winner_driver = next(
        (d for d in drivers if d.get("driver_number") == winner_number),
        {"full_name": f"Driver #{winner_number}"},
    )

    # Get stints and pits for winner
    winner_stints = await client.get_stints(session_key=session_key, driver_number=winner_number)
    winner_pits = await client.get_pit(session_key=session_key, driver_number=winner_number)

    # Weather summary
    weather_data = await client.get_weather(session_key=session_key)
    weather_summary = _summarise_weather(weather_data)

    # Race control events (safety cars, flags)
    rc_messages = await client.get_race_control(session_key=session_key)
    key_events = _summarise_race_control(rc_messages)

    # Determine total laps from position data
    total_laps = max((p.get("lap", 0) or 0) for p in positions) if positions else 0
    session["laps"] = total_laps or session.get("laps", 0)

    # Build strategy object (no summary yet)
    strategy = build_historical_strategy(
        session=session,
        winner_driver=winner_driver,
        winner_stints=winner_stints,
        winner_pits=winner_pits,
        weather_summary=weather_summary,
        key_events=key_events,
        summary="",
    )

    # Generate LLM summary
    try:
        summary = generate_strategy_summary(strategy)
    except Exception as exc:
        logger.warning("LLM summary failed for %s: %s", session.get("meeting_name"), exc)
        summary = f"{strategy.winner} won with {strategy.winner_strategy} strategy."

    return strategy.model_copy(update={"summary": summary})


def _summarise_weather(weather_data: list[dict]) -> str:
    if not weather_data:
        return "Unknown conditions"
    avg_air = sum(w.get("air_temperature", 0) or 0 for w in weather_data) / len(weather_data)
    avg_track = sum(w.get("track_temperature", 0) or 0 for w in weather_data) / len(weather_data)
    rainfall = any(w.get("rainfall") for w in weather_data)
    rain_str = "Rain during race" if rainfall else "Dry"
    return f"{rain_str}. Avg air {avg_air:.0f}°C, track {avg_track:.0f}°C."


def _summarise_race_control(messages: list[dict]) -> str:
    events = []
    for msg in messages:
        text = msg.get("message", "").lower()
        if any(kw in text for kw in ("safety car", "virtual safety car", "red flag")):
            events.append(msg.get("message", ""))
    if not events:
        return "No notable events"
    return "; ".join(events[:5])  # cap at 5 events


def main() -> None:
    parser = argparse.ArgumentParser(description="Index F1 race strategies into Qdrant")
    parser.add_argument("--year", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--dry-run", action="store_true", help="Print without indexing")
    args = parser.parse_args()

    asyncio.run(run(years=args.year, dry_run=args.dry_run))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Smoke-test script: verify OpenF1 API works end-to-end.

Run from the repo root:
    python scripts/test_openf1.py

This makes real network requests and populates the SQLite cache.
On the second run everything is served from cache (no network calls).
"""

import asyncio
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logging import configure_logging
from src.data.openf1_client import OpenF1Client
from src.data.race_builder import build_race_state


async def main() -> None:
    configure_logging()

    print("\n=== Pitwall-AI: OpenF1 smoke test ===\n")

    async with OpenF1Client() as client:

        # ------------------------------------------------------------------
        # 1. List 2024 race sessions
        # ------------------------------------------------------------------
        print("1. Fetching 2024 race sessions...")
        sessions = await client.get_sessions(year=2024, session_type="Race")
        print(f"   Found {len(sessions)} race sessions in 2024")
        if not sessions:
            print("   ERROR: no sessions returned — check your network/API")
            return

        for s in sessions[:3]:
            print(f"   • {s.get('meeting_name')} — session_key={s.get('session_key')}")

        # Use the first available race for deeper tests
        session = sessions[0]
        session_key: int = int(session["session_key"])
        print(f"\n   Using session_key={session_key} ({session.get('meeting_name')}) for remaining tests\n")

        # ------------------------------------------------------------------
        # 2. Drivers
        # ------------------------------------------------------------------
        print("2. Fetching drivers...")
        drivers = await client.get_drivers(session_key)
        print(f"   Found {len(drivers)} drivers")
        for d in drivers[:3]:
            print(f"   • #{d.get('driver_number')} {d.get('full_name')} — {d.get('team_name')}")

        # ------------------------------------------------------------------
        # 3. Stints (tire compounds)
        # ------------------------------------------------------------------
        print("\n3. Fetching stints...")
        stints = await client.get_stints(session_key)
        print(f"   Found {len(stints)} stint records")
        for s in stints[:3]:
            print(
                f"   • Driver #{s.get('driver_number')} | {s.get('compound')} | "
                f"laps {s.get('lap_start')}–{s.get('lap_end')}"
            )

        # ------------------------------------------------------------------
        # 4. Pit stops
        # ------------------------------------------------------------------
        print("\n4. Fetching pit stops...")
        pits = await client.get_pit(session_key)
        print(f"   Found {len(pits)} pit stop records")
        for p in pits[:3]:
            print(
                f"   • Driver #{p.get('driver_number')} | lap {p.get('lap_number')} | "
                f"{p.get('pit_duration')}s"
            )

        # ------------------------------------------------------------------
        # 5. Weather
        # ------------------------------------------------------------------
        print("\n5. Fetching weather...")
        weather = await client.get_weather(session_key)
        if weather:
            w = weather[-1]
            print(
                f"   Air {w.get('air_temperature')}°C | "
                f"Track {w.get('track_temperature')}°C | "
                f"Rain={w.get('rainfall')}"
            )

        # ------------------------------------------------------------------
        # 6. Build full RaceState (tests race_builder)
        # ------------------------------------------------------------------
        print("\n6. Building full RaceState (may take a moment on first run)...")
        try:
            race_state = await build_race_state(client, session_key)
            print(f"   session_key : {race_state.session_key}")
            print(f"   meeting     : {race_state.meeting_name}")
            print(f"   track       : {race_state.track_name}")
            print(f"   laps        : {race_state.current_lap}/{race_state.total_laps}")
            print(f"   drivers     : {len(race_state.drivers)}")
            print(f"   weather     : {race_state.weather}")
            print(f"   rc messages : {len(race_state.race_control)}")
            print()
            for d in race_state.drivers[:5]:
                print(
                    f"   P{d.position:2d}  #{d.driver_number:2d} {d.name:<25} "
                    f"{d.tire_compound.value:<12} "
                    f"stint_laps={d.stint_length}  "
                    f"gap={d.gap_to_leader}"
                )
        except Exception as exc:
            print(f"   ERROR building RaceState: {exc}")
            raise

        # ------------------------------------------------------------------
        # 7. Replay mode — RaceState at lap 20
        # ------------------------------------------------------------------
        print("\n7. Building RaceState at lap 20 (replay mode)...")
        try:
            race_at_20 = await build_race_state(client, session_key, up_to_lap=20)
            print(f"   current_lap={race_at_20.current_lap}, drivers={len(race_at_20.drivers)}")
        except Exception as exc:
            print(f"   ERROR: {exc}")

    print("\n=== Smoke test complete ===\n")


if __name__ == "__main__":
    asyncio.run(main())

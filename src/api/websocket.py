"""WebSocket endpoint for lap-by-lap race replay.

Clients connect to /ws/race/{session_key} and receive a stream of
JSON-encoded RaceState snapshots, one per lap, with a configurable delay.

Protocol (server → client):
    {"type": "lap", "lap": 12, "data": <RaceState.model_dump()>}
    {"type": "done", "message": "Replay complete"}
    {"type": "error", "message": "..."}

Client can send:
    {"action": "pause"}   — pause the replay
    {"action": "resume"}  — resume the replay
    {"action": "stop"}    — end the session
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.data.openf1_client import OpenF1Client
from src.data.race_builder import RaceBuilder

logger = logging.getLogger(__name__)

router = APIRouter()

# Delay between laps during replay (seconds)
DEFAULT_LAP_DELAY_S: float = 1.0
MIN_LAP_DELAY_S: float = 0.1
MAX_LAP_DELAY_S: float = 10.0


@router.websocket("/ws/race/{session_key}")
async def race_replay(
    websocket: WebSocket,
    session_key: int,
    delay: float = DEFAULT_LAP_DELAY_S,
) -> None:
    """Stream lap-by-lap race state snapshots to the client.

    Query params:
        delay (float):  Seconds between each lap update (default 1.0).
    """
    await websocket.accept()
    logger.info("WebSocket connection opened: session_key=%s", session_key)

    # Clamp delay to safe range
    delay = max(MIN_LAP_DELAY_S, min(delay, MAX_LAP_DELAY_S))

    paused = False
    stopped = False

    async def listen_for_commands() -> None:
        """Background task: listen for pause/resume/stop commands."""
        nonlocal paused, stopped
        try:
            while not stopped:
                msg = await websocket.receive_json()
                action = msg.get("action", "")
                if action == "pause":
                    paused = True
                    logger.debug("Replay paused")
                elif action == "resume":
                    paused = False
                    logger.debug("Replay resumed")
                elif action == "stop":
                    stopped = True
                    logger.debug("Replay stopped by client")
        except WebSocketDisconnect:
            stopped = True
        except Exception as exc:
            logger.warning("WebSocket command listener error: %s", exc)
            stopped = True

    # Start the command listener as a background task
    listener_task = asyncio.create_task(listen_for_commands())

    try:
        async with OpenF1Client() as client:
            # Determine total laps first
            builder = RaceBuilder(client)
            full_state = await builder.build(session_key=session_key)
            total_laps = full_state.total_laps

            if total_laps == 0:
                await websocket.send_json(
                    {"type": "error", "message": "No lap data found for this session"}
                )
                return

            # Stream lap by lap
            for lap in range(1, total_laps + 1):
                if stopped:
                    break

                # Wait while paused
                while paused and not stopped:
                    await asyncio.sleep(0.2)

                if stopped:
                    break

                try:
                    state = await builder.build(session_key=session_key, up_to_lap=lap)
                    await websocket.send_json(
                        {
                            "type": "lap",
                            "lap": lap,
                            "total_laps": total_laps,
                            "data": state.model_dump(mode="json"),
                        }
                    )
                    logger.debug("Sent lap %d/%d to WebSocket client", lap, total_laps)
                except Exception as exc:
                    logger.warning("Failed to build state at lap %d: %s", lap, exc)
                    await websocket.send_json(
                        {"type": "error", "message": f"Lap {lap} data unavailable: {exc}"}
                    )

                await asyncio.sleep(delay)

            if not stopped:
                await websocket.send_json(
                    {"type": "done", "message": f"Replay complete — {total_laps} laps"}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session_key=%s", session_key)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass
        logger.info("WebSocket session closed: session_key=%s", session_key)

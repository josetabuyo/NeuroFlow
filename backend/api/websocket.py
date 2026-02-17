"""WebSocket handler for NeuroFlow experiments."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from experiments.base import Experimento
from experiments.von_neumann import VonNeumannExperiment
from experiments.kohonen import KohonenExperiment
from experiments.kohonen_balanced import KohonenBalancedExperiment

logger = logging.getLogger(__name__)

ws_router = APIRouter()

EXPERIMENT_CLASSES: dict[str, type[Experimento]] = {
    "von_neumann": VonNeumannExperiment,
    "kohonen": KohonenExperiment,
    "kohonen_balanced": KohonenBalancedExperiment,
}


class ExperimentSession:
    """Manages a single WebSocket experiment session."""

    def __init__(self, websocket: WebSocket) -> None:
        self.ws = websocket
        self.experiment: Experimento | None = None
        self._play_task: asyncio.Task | None = None
        self._playing: bool = False
        self.fps: int = 10
        self.steps_per_tick: int = 1

    async def send(self, data: dict[str, Any]) -> None:
        """Send JSON data to the client."""
        await self.ws.send_json(data)

    async def handle_message(self, message: dict[str, Any]) -> None:
        """Route incoming messages to the appropriate handler."""
        action = message.get("action", "")

        handlers = {
            "start": self._handle_start,
            "click": self._handle_click,
            "paint": self._handle_paint,
            "step": self._handle_step,
            "play": self._handle_play,
            "pause": self._handle_pause,
            "reset": self._handle_reset,
            "inspect": self._handle_inspect,
        }

        handler = handlers.get(action)
        if handler:
            await handler(message)
        else:
            await self.send({"type": "error", "message": f"Unknown action: {action}"})

    async def _handle_start(self, message: dict[str, Any]) -> None:
        """Initialize an experiment."""
        experiment_id = message.get("experiment", "von_neumann")
        config = message.get("config", {})

        exp_class = EXPERIMENT_CLASSES.get(experiment_id)
        if not exp_class:
            await self.send({"type": "error", "message": f"Unknown experiment: {experiment_id}"})
            return

        self.experiment = exp_class()
        self.experiment.setup(config)

        await self.send({"type": "status", "state": "ready"})
        await self._send_frame()

    async def _handle_click(self, message: dict[str, Any]) -> None:
        """Handle a click on the canvas."""
        if not self.experiment:
            await self.send({"type": "error", "message": "No experiment started"})
            return

        x = message.get("x", 0)
        y = message.get("y", 0)
        self.experiment.click(x, y)
        await self._send_frame()

    async def _handle_paint(self, message: dict[str, Any]) -> None:
        """Apply brush: activate/deactivate multiple neurons at once."""
        if not self.experiment:
            await self.send({"type": "error", "message": "No experiment started"})
            return
        cells: list[dict[str, int]] = message.get("cells", [])
        value: float = message.get("value", 1.0)

        red_tensor = getattr(self.experiment, "red_tensor", None)
        if red_tensor:
            w = self.experiment.width
            for cell in cells:
                x = cell.get("x", 0)
                y = cell.get("y", 0)
                idx = y * w + x
                if 0 <= x < w and 0 <= y < self.experiment.height and 0 <= idx < red_tensor.n_real:
                    red_tensor.set_valor(idx, value)
        await self._send_frame()

    async def _handle_step(self, message: dict[str, Any]) -> None:
        """Process one or more steps. Supports {"action": "step", "count": N}."""
        if not self.experiment:
            await self.send({"type": "error", "message": "No experiment started"})
            return

        count = max(1, message.get("count", 1))

        t0 = time.perf_counter()
        result = self.experiment.step_n(count)
        elapsed = time.perf_counter() - t0

        if result.get("type") == "status" and result.get("state") == "complete":
            await self.send(result)
            return

        await self._send_frame(steps=count, elapsed_s=elapsed)

    async def _handle_play(self, message: dict[str, Any]) -> None:
        """Start continuous processing."""
        if not self.experiment:
            await self.send({"type": "error", "message": "No experiment started"})
            return

        self.fps = message.get("fps", 10)
        self.steps_per_tick = max(1, message.get("steps_per_tick", 1))
        self._playing = True
        await self.send({"type": "status", "state": "running"})

        if self._play_task and not self._play_task.done():
            self._play_task.cancel()

        self._play_task = asyncio.create_task(self._play_loop())

    async def _handle_pause(self, _message: dict[str, Any]) -> None:
        """Pause continuous processing."""
        self._playing = False
        if self._play_task and not self._play_task.done():
            self._play_task.cancel()
        await self.send({"type": "status", "state": "paused"})

    async def _handle_inspect(self, message: dict[str, Any]) -> None:
        """Return connection weights for a neuron."""
        if not self.experiment:
            await self.send({"type": "error", "message": "No experiment started"})
            return
        x = message.get("x", 0)
        y = message.get("y", 0)
        result = self.experiment.inspect(x, y)
        await self.send(result)

    async def _handle_reset(self, _message: dict[str, Any]) -> None:
        """Reset the experiment."""
        if not self.experiment:
            await self.send({"type": "error", "message": "No experiment started"})
            return

        self._playing = False
        if self._play_task and not self._play_task.done():
            self._play_task.cancel()

        self.experiment.reset()
        await self.send({"type": "status", "state": "ready"})
        await self._send_frame()

    async def _play_loop(self) -> None:
        """Continuously process and send frames."""
        try:
            while self._playing and self.experiment:
                t0 = time.perf_counter()
                result = self.experiment.step_n(self.steps_per_tick)
                elapsed = time.perf_counter() - t0

                if result.get("type") == "status" and result.get("state") == "complete":
                    await self.send(result)
                    self._playing = False
                    return
                await self._send_frame(steps=self.steps_per_tick, elapsed_s=elapsed)
                await asyncio.sleep(1.0 / self.fps)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Error in play loop")
            await self.send({"type": "error", "message": str(e)})

    async def _send_frame(
        self,
        steps: int | None = None,
        elapsed_s: float | None = None,
    ) -> None:
        """Send the current frame to the client, with optional timing metrics."""
        if not self.experiment:
            return

        frame = self.experiment.get_frame()
        stats = self.experiment.get_stats()

        grid = [[round(cell) for cell in row] for row in frame]

        msg: dict[str, Any] = {
            "type": "frame",
            "generation": self.experiment.generation,
            "grid": grid,
            "stats": stats,
        }

        if steps is not None and elapsed_s is not None and elapsed_s > 0:
            msg["perf"] = {
                "steps": steps,
                "elapsed_ms": round(elapsed_s * 1000, 2),
                "steps_per_second": round(steps / elapsed_s, 1),
            }

        await self.send(msg)

    def cleanup(self) -> None:
        """Cleanup on disconnect."""
        self._playing = False
        if self._play_task and not self._play_task.done():
            self._play_task.cancel()


@ws_router.websocket("/ws/experiment")
async def experiment_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for experiment interaction."""
    await websocket.accept()
    session = ExperimentSession(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                await session.handle_message(message)
            except json.JSONDecodeError:
                await session.send({"type": "error", "message": "Invalid JSON"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
    finally:
        session.cleanup()

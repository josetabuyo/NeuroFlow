"""Deamons Lab experiment — configurable mask connectivity lab.

Choose from multiple wiring presets (Mexican hat) and adjust the
excitation/inhibition balance. Supports hot reconnection: change mask
and balance without losing neuron state.

Includes "daemon" metrics (Dennett): bell-shaped activation clusters
that compete through lateral exclusion.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import torch

from core.constructor import Constructor
from core.constructor_tensor import ConstructorTensor
from core.masks import get_mask, get_mask_type, get_random_weights
from .base import Experimento

_STABILITY_WINDOW = 20
_DAEMON_THRESHOLD = 0.5
_MIN_DAEMON_SIZE = 3


class _DaemonResult:
    """Result of daemon detection: clusters, noise, sizes."""

    __slots__ = ("count", "daemon_indices", "noise_indices", "sizes")

    def __init__(
        self,
        count: int,
        daemon_indices: set[int],
        noise_indices: set[int],
        sizes: list[int],
    ) -> None:
        self.count = count
        self.daemon_indices = daemon_indices
        self.noise_indices = noise_indices
        self.sizes = sizes


def _detect_daemons(
    values: torch.Tensor,
    width: int,
    height: int,
    threshold: float,
    min_size: int = _MIN_DAEMON_SIZE,
) -> _DaemonResult:
    """Detect daemons as connected components of active neurons (8-connectivity).

    Only clusters with >= min_size neurons count as daemons.
    Smaller active groups are classified as noise.
    """
    n = width * height
    active = (values[:n] > threshold).tolist()
    visited = [False] * n
    daemon_indices: set[int] = set()
    noise_indices: set[int] = set()
    sizes: list[int] = []

    for idx in range(n):
        if active[idx] and not visited[idx]:
            queue = deque([idx])
            visited[idx] = True
            cluster: list[int] = []
            while queue:
                cidx = queue.popleft()
                cluster.append(cidx)
                cx, cy = cidx % width, cidx // width
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            nidx = ny * width + nx
                            if active[nidx] and not visited[nidx]:
                                visited[nidx] = True
                                queue.append(nidx)
            if len(cluster) >= min_size:
                daemon_indices.update(cluster)
                sizes.append(len(cluster))
            else:
                noise_indices.update(cluster)

    return _DaemonResult(
        count=len(sizes),
        daemon_indices=daemon_indices,
        noise_indices=noise_indices,
        sizes=sizes,
    )


class DeamonsLabExperiment(Experimento):
    """Configurable mask connectivity lab."""

    def __init__(self) -> None:
        super().__init__()
        self._config: dict[str, Any] = {}
        self.red_tensor = None
        self._daemon_history: deque[int] = deque(maxlen=_STABILITY_WINDOW)
        self._last_history_gen: int = -1

    def setup(self, config: dict[str, Any]) -> None:
        """Configure a 2D grid with the chosen mask, balance, and initialization.

        Config keys:
            width (int): Grid width (default 50).
            height (int): Grid height (default 50).
            mask (str): Mask preset ID (default "simple").
            balance (float | None): Excitation/inhibition balance (default None).
            balance_mode (str): "none", "weight", or "synapse_count".

        Wolfram masks (mask_type == "wolfram") automatically set:
            umbral=0.99, bottom row as input, single center cell initialization.
        """
        self._config = config
        self.width = config.get("width", 50)
        self.height = config.get("height", 50)
        self.generation = 0

        mask_id: str = config.get("mask", "simple")
        balance = config.get("balance", None)
        balance_mode: str = config.get("balance_mode", "none")
        mask = get_mask(mask_id)
        self._mask_type = get_mask_type(mask_id)
        self._random_weights = get_random_weights(mask_id)

        constructor = Constructor()

        is_wolfram = self._mask_type == "wolfram"

        self.red, self.regiones = constructor.crear_grilla(
            width=self.width,
            height=self.height,
            filas_entrada=[self.height - 1] if is_wolfram else [],
            filas_salida=[],
            umbral=0.99 if is_wolfram else 0.0,
        )

        constructor.aplicar_mascara_2d(
            self.red, self.width, self.height, mask,
            random_weights=self._random_weights,
        )

        if balance is not None and balance_mode == "weight":
            constructor.balancear_pesos(
                list(self.red.neuronas), target=balance
            )
        elif balance is not None and balance_mode == "synapse_count":
            constructor.balancear_sinapsis(
                list(self.red.neuronas), target=balance
            )

        if is_wolfram:
            for neurona in self.red.neuronas:
                neurona.activar_external(0.0)
            center_x = self.width // 2
            bottom_y = self.height - 1
            self.red.get_neurona(
                f"x{center_x}y{bottom_y}"
            ).activar_external(1.0)
        else:
            for neurona in self.red.neuronas:
                neurona.activar_external(random.random())

        self.red_tensor = ConstructorTensor.compilar(self.red)

        self._daemon_history.clear()
        self._last_history_gen = -1

    def reconnect(self, config: dict[str, Any]) -> None:
        """Change mask and/or balance. Wolfram masks do a full reset (new init)."""
        if self.red_tensor is None:
            return

        mask_id = config.get("mask", self._config.get("mask", "simple"))
        new_mask_type = get_mask_type(mask_id)

        if new_mask_type != self._mask_type:
            self._config["mask"] = mask_id
            self._config["balance"] = config.get(
                "balance", self._config.get("balance", None)
            )
            self._config["balance_mode"] = config.get(
                "balance_mode", self._config.get("balance_mode", "none")
            )
            self.setup(self._config)
            return

        saved_values = self.red_tensor.valores[: self.red_tensor.n_real].clone()

        balance = config.get("balance", self._config.get("balance", None))
        balance_mode = config.get(
            "balance_mode", self._config.get("balance_mode", "none")
        )
        self._config["mask"] = mask_id
        self._config["balance"] = balance
        self._config["balance_mode"] = balance_mode

        mask = get_mask(mask_id)
        self._random_weights = get_random_weights(mask_id)
        is_wolfram = self._mask_type == "wolfram"
        constructor = Constructor()

        self.red, self.regiones = constructor.crear_grilla(
            width=self.width,
            height=self.height,
            filas_entrada=[self.height - 1] if is_wolfram else [],
            filas_salida=[],
            umbral=0.99 if is_wolfram else 0.0,
        )

        constructor.aplicar_mascara_2d(
            self.red, self.width, self.height, mask,
            random_weights=self._random_weights,
        )

        if balance is not None and balance_mode == "weight":
            constructor.balancear_pesos(
                list(self.red.neuronas), target=balance
            )
        elif balance is not None and balance_mode == "synapse_count":
            constructor.balancear_sinapsis(
                list(self.red.neuronas), target=balance
            )

        for i, neurona in enumerate(self.red.neuronas):
            neurona.activar_external(saved_values[i].item())

        self.red_tensor = ConstructorTensor.compilar(self.red)

        self._daemon_history.clear()
        self._last_history_gen = -1

    def click(self, x: int, y: int) -> None:
        """Toggle: if value < 0.5 -> activate (1.0), if >= 0.5 -> deactivate (0.0)."""
        idx = y * self.width + x
        if 0 <= idx < self.red_tensor.n_real:
            current = self.red_tensor.valores[idx].item()
            self.red_tensor.set_valor(idx, 0.0 if current >= 0.5 else 1.0)

    def step(self) -> dict[str, Any]:
        """One step processes the entire network at once."""
        self.red_tensor.procesar()
        self.generation += 1

        frame = self.get_frame()
        stats = self.get_stats()

        return {
            "type": "frame",
            "generation": self.generation,
            "grid": frame,
            "stats": stats,
        }

    def step_n(self, count: int) -> dict[str, Any]:
        """N steps in a single tensor operation."""
        self.red_tensor.procesar_n(count)
        self.generation += count
        return {
            "type": "frame",
            "generation": self.generation,
            "grid": self.get_frame(),
            "stats": self.get_stats(),
        }

    def get_frame(self) -> list[list[float]]:
        """Return the current grid as a matrix of values."""
        if self.red_tensor:
            return self.red_tensor.get_grid(self.width, self.height)
        return super().get_frame()

    def get_tension_frame(self) -> list[list[float]] | None:
        """Return the surface tension grid."""
        if self.red_tensor:
            return self.red_tensor.get_tension_grid(self.width, self.height)
        return None

    def get_stats(self) -> dict[str, Any]:
        """Return statistics with daemon metrics.

        Daemon metrics (only clusters >= _MIN_DAEMON_SIZE count as daemons):
        - daemon_count: number of daemons (real clusters, not noise)
        - avg_daemon_size: average neurons per daemon
        - noise_cells: active neurons NOT in any daemon (isolated/small groups)
        - exclusion: mean activation inside daemons minus outside (higher = better)
        - stability: 1 - CV of daemon count over a sliding window (higher = more stable)
        """
        if self.red_tensor is None:
            return super().get_stats()

        vals = self.red_tensor.valores[: self.width * self.height]
        n = self.width * self.height

        result = _detect_daemons(
            self.red_tensor.valores, self.width, self.height, _DAEMON_THRESHOLD
        )

        active = int((vals > _DAEMON_THRESHOLD).sum().item())
        avg_size = round(sum(result.sizes) / len(result.sizes), 1) if result.sizes else 0.0

        if result.daemon_indices:
            daemon_mask = torch.zeros(n, dtype=torch.bool)
            daemon_mask[list(result.daemon_indices)] = True
            inside_mean = vals[daemon_mask].mean().item()
            outside = vals[~daemon_mask]
            outside_mean = outside.mean().item() if outside.numel() > 0 else 0.0
            exclusion = inside_mean - outside_mean
        else:
            exclusion = 0.0

        if self.generation != self._last_history_gen:
            self._daemon_history.append(result.count)
            self._last_history_gen = self.generation

        if len(self._daemon_history) >= 2:
            counts = list(self._daemon_history)
            mean_c = sum(counts) / len(counts)
            if mean_c > 0:
                variance = sum((c - mean_c) ** 2 for c in counts) / len(counts)
                cv = (variance ** 0.5) / mean_c
                stability = round(max(0.0, min(1.0, 1.0 - cv)), 3)
            else:
                stability = 1.0 if all(c == 0 for c in counts) else 0.0
        else:
            stability = 0.0

        return {
            "active_cells": active,
            "steps": self.generation,
            "daemon_count": result.count,
            "avg_daemon_size": avg_size,
            "noise_cells": len(result.noise_indices),
            "stability": stability,
            "exclusion": round(exclusion, 3),
        }

    def reset(self) -> None:
        """Reset the experiment with the same configuration."""
        self.setup(self._config)

    def is_complete(self) -> bool:
        """Deamons Lab never ends."""
        return False

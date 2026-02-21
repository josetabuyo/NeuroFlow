"""Base class for NeuroFlow experiments."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from core.constructor import Constructor
from core.red import Red
from core.region import Region


def _parse_coords(neuron_id: str) -> tuple[int, int] | None:
    """Extract (x, y) from neuron ID. Returns None if not parseable."""
    match = re.match(r"^x(\d+)y(\d+)$", neuron_id)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


class Experimento(ABC):
    """Base class for experiments.

    Orchestrates: what is input, what is output,
    how it is fed, how it is read.
    """

    def __init__(self) -> None:
        self.red: Red | None = None
        self.regiones: dict[str, Region] = {}
        self.width: int = 0
        self.height: int = 0
        self.generation: int = 0

    @abstractmethod
    def setup(self, config: dict[str, Any]) -> None:
        """Use Constructor to build Red + Regions."""
        ...

    @abstractmethod
    def step(self) -> dict[str, Any]:
        """Advance one processing step."""
        ...

    def step_n(self, count: int) -> dict[str, Any]:
        """Advance N processing steps efficiently.

        Subclasses should override for optimized bulk processing.
        Default falls back to calling step() in a loop.
        """
        result: dict[str, Any] = {}
        for _ in range(count):
            result = self.step()
            if result.get("type") == "status" and result.get("state") == "complete":
                return result
        return result

    @abstractmethod
    def click(self, x: int, y: int) -> None:
        """Activate a neuron in the input region."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the experiment."""
        ...

    def get_frame(self) -> list[list[float]]:
        """Return the current grid as a value matrix."""
        if self.red is None:
            return []
        return self.red.get_grid(self.width, self.height)

    def get_tension_frame(self) -> list[list[float]] | None:
        """Return the tension grid. None if not available."""
        return None

    def get_stats(self) -> dict[str, Any]:
        """Return statistics for the current state."""
        frame = self.get_frame()
        active = sum(1 for row in frame for cell in row if cell > 0)
        return {
            "active_cells": active,
            "steps": self.generation,
        }

    def inspect(self, x: int, y: int) -> dict[str, Any]:
        """Return the effective weight map for a neuron.

        For each synapse of the neuron at (x, y), computes:
          effective_weight = synapse.peso × dendrite.peso

        If a source neuron appears in multiple dendrites, sums the
        effective weights. Clamps the result to [-1, 1].

        Returns:
            {
                "type": "connections",
                "x": int,
                "y": int,
                "total_dendritas": int,
                "total_sinapsis": int,
                "weight_grid": list[list[float | None]]
            }
            weight_grid contains:
            - float in [-1, 1] for connected cells (effective weight)
            - None for unconnected cells
            - The inspected cell is marked with special value 999.
        """
        if self.red is None:
            return {
                "type": "connections",
                "x": x,
                "y": y,
                "total_dendritas": 0,
                "total_sinapsis": 0,
                "weight_grid": [],
            }

        key = Constructor.key_by_coord(x, y)
        neurona = self.red.get_neurona(key)

        # Accumulate effective weights by source neuron
        pesos: dict[str, float] = {}
        total_sinapsis = 0

        for dendrita in neurona.dendritas:
            for sinapsis in dendrita.sinapsis:
                total_sinapsis += 1
                fuente_id = sinapsis.neurona_entrante.id
                if fuente_id.startswith("_borde_"):
                    continue
                peso_efectivo = sinapsis.peso * dendrita.peso
                pesos[fuente_id] = pesos.get(fuente_id, 0.0) + peso_efectivo

        # Clamp to [-1, 1]
        for nid in pesos:
            if pesos[nid] > 1.0:
                pesos[nid] = 1.0
            elif pesos[nid] < -1.0:
                pesos[nid] = -1.0

        # Build weight_grid
        weight_grid: list[list[float | None]] = []
        for row in range(self.height):
            fila: list[float | None] = []
            for col in range(self.width):
                if col == x and row == y:
                    fila.append(999)
                else:
                    cell_key = Constructor.key_by_coord(col, row)
                    if cell_key in pesos:
                        fila.append(pesos[cell_key])
                    else:
                        fila.append(None)
            weight_grid.append(fila)

        return {
            "type": "connections",
            "x": x,
            "y": y,
            "total_dendritas": len(neurona.dendritas),
            "total_sinapsis": total_sinapsis,
            "weight_grid": weight_grid,
        }

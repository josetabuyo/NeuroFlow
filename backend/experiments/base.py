"""Base class for NeuroFlow experiments."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from core.constructor import Constructor
from core.red import Red
from core.region import Region


def _parse_coords(neuron_id: str) -> tuple[int, int] | None:
    """Extrae (x, y) del ID de neurona. Retorna None si no es parseable."""
    match = re.match(r"^x(\d+)y(\d+)$", neuron_id)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


class Experimento(ABC):
    """Clase base para experimentos.

    Orquesta: qué es entrada, qué es salida,
    cómo se alimenta, cómo se lee.
    """

    def __init__(self) -> None:
        self.red: Red | None = None
        self.regiones: dict[str, Region] = {}
        self.width: int = 0
        self.height: int = 0
        self.generation: int = 0

    @abstractmethod
    def setup(self, config: dict[str, Any]) -> None:
        """Usa Constructor para armar Red + Regiones."""
        ...

    @abstractmethod
    def step(self) -> dict[str, Any]:
        """Avanza un paso de procesamiento."""
        ...

    def step_n(self, count: int) -> dict[str, Any]:
        """Avanza N pasos de procesamiento de forma eficiente.

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
        """Activa una neurona en la región de entrada."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reinicia el experimento."""
        ...

    def get_frame(self) -> list[list[float]]:
        """Retorna la grilla actual como matriz de valores."""
        if self.red is None:
            return []
        return self.red.get_grid(self.width, self.height)

    def get_stats(self) -> dict[str, Any]:
        """Retorna estadísticas del estado actual."""
        frame = self.get_frame()
        active = sum(1 for row in frame for cell in row if cell > 0)
        return {
            "active_cells": active,
            "steps": self.generation,
        }

    def inspect(self, x: int, y: int) -> dict[str, Any]:
        """Retorna el mapa de pesos efectivos para una neurona.

        Para cada sinapsis de la neurona en (x, y), calcula:
          peso_efectivo = sinapsis.peso × dendrita.peso

        Si una neurona fuente aparece en múltiples dendritas, suma los
        pesos efectivos. Clampea el resultado a [-1, 1].

        Returns:
            {
                "type": "connections",
                "x": int,
                "y": int,
                "total_dendritas": int,
                "total_sinapsis": int,
                "weight_grid": list[list[float | None]]
            }
            weight_grid contiene:
            - float en [-1, 1] para celdas conectadas (peso efectivo)
            - None para celdas sin conexión
            - La celda inspeccionada se marca con valor especial 999.
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

        # Acumular pesos efectivos por neurona fuente
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

        # Clampear a [-1, 1]
        for nid in pesos:
            if pesos[nid] > 1.0:
                pesos[nid] = 1.0
            elif pesos[nid] < -1.0:
                pesos[nid] = -1.0

        # Construir weight_grid
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

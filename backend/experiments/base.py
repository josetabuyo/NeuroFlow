"""Base class for NeuroFlow experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.red import Red
from core.region import Region


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
            "generation": self.generation,
            "total_rows": self.height,
        }

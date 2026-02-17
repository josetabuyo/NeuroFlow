"""Experimento Von Neumann — autómata celular elemental (1D, reglas de Wolfram).

Usa Constructor para armar:
- Red de width x height neuronas
- Region "entrada" = fila inferior (NeuronaEntrada)
- Region "salida" = fila superior
- Region "interna" = filas intermedias

Constructor conecta cada neurona interna/salida a 3 de la fila inferior.
Constructor configura dendritas según tabla de la regla.

Procesamiento: cada step() llama a procesar() sobre toda la red.
Cada procesar() propaga el "frente de onda" exactamente una fila hacia arriba
(porque las filas superiores leen de filas inferiores todavía no actualizadas).
Después de height-1 steps, el autómata completo está calculado.
"""

from __future__ import annotations

from typing import Any

from core.constructor import Constructor
from core.constructor_tensor import ConstructorTensor
from .base import Experimento


class VonNeumannExperiment(Experimento):
    """Autómata celular elemental implementado con neuronas."""

    def __init__(self) -> None:
        super().__init__()
        self.rule: int = 111
        self._current_row: int = 0
        self._config: dict[str, Any] = {}
        self.red_tensor = None

    def setup(self, config: dict[str, Any]) -> None:
        """Configura el experimento con la regla y dimensiones dadas."""
        self._config = config
        self.width = config.get("width", 50)
        self.height = config.get("height", 50)
        self.rule = config.get("rule", 111)
        self.generation = 0

        constructor = Constructor()

        fila_entrada = self.height - 1
        fila_salida = 0

        self.red, self.regiones = constructor.crear_grilla(
            width=self.width,
            height=self.height,
            filas_entrada=[fila_entrada],
            filas_salida=[fila_salida],
            umbral=0.99,
        )

        for fila in range(self.height - 2, -1, -1):
            constructor.aplicar_regla_wolfram(
                red=self.red,
                regla=self.rule,
                fila_destino=fila,
                width=self.width,
            )

        self._current_row = self.height - 2

        self.red_tensor = ConstructorTensor.compilar(self.red)

    def click(self, x: int, y: int) -> None:
        """Activa una neurona en la fila de entrada."""
        fila_entrada = self.height - 1
        if y == fila_entrada:
            idx = y * self.width + x
            if 0 <= idx < self.red_tensor.n_real:
                self.red_tensor.set_valor(idx, 1.0)

    def get_stats(self) -> dict[str, Any]:
        """Von Neumann tiene un total conocido de steps (height - 1)."""
        stats = super().get_stats()
        stats["total_steps"] = self.height - 1
        return stats

    def step(self) -> dict[str, Any]:
        """Procesa un step del autómata.

        Cada procesar() propaga el frente de onda una fila hacia arriba.
        Filas ya calculadas se recalculan (idempotente).
        """
        if self._current_row < 0:
            return {"type": "status", "state": "complete"}

        self.red_tensor.procesar()

        self._current_row -= 1
        self.generation += 1

        return {
            "type": "frame",
            "generation": self.generation,
            "grid": self.get_frame(),
            "stats": self.get_stats(),
        }

    def step_n(self, count: int) -> dict[str, Any]:
        """N steps en una sola operación tensorial, capeado a filas restantes."""
        if self._current_row < 0:
            return {"type": "status", "state": "complete"}

        remaining = self._current_row + 1
        actual = min(count, remaining)

        self.red_tensor.procesar_n(actual)
        self._current_row -= actual
        self.generation += actual

        frame = self.get_frame()
        stats = self.get_stats()

        if self._current_row < 0:
            return {
                "type": "status",
                "state": "complete",
                "grid": frame,
                "stats": stats,
                "generation": self.generation,
            }

        return {
            "type": "frame",
            "generation": self.generation,
            "grid": frame,
            "stats": stats,
        }

    def get_frame(self) -> list[list[float]]:
        """Retorna la grilla actual como matriz de valores."""
        if self.red_tensor:
            return self.red_tensor.get_grid(self.width, self.height)
        return super().get_frame()

    def reset(self) -> None:
        """Reinicia el experimento con la misma configuración."""
        self.setup(self._config)

    def is_complete(self) -> bool:
        """Retorna True si ya se procesaron todas las filas."""
        return self._current_row < 0

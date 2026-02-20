"""Experimento Kohonen — competencia lateral 2D (mapa autoorganizado).

Implementa el conexionado `kohonen_simple` del modelo original:
- 1 dendrita excitatoria (vecinos inmediatos, Moore neighborhood)
- 8 dendritas inhibitorias (bloques 3×3 a distancia ~3, anillo octagonal)

Cada neurona excita a sus vecinas cercanas e inhibe a las lejanas.
El perfil "Mexican hat" produce clusters de actividad que compiten entre sí.
"""

from __future__ import annotations

import random
from typing import Any

from core.constructor import Constructor
from core.constructor_tensor import ConstructorTensor
from core.masks import MASK_SIMPLE
from .base import Experimento


# Re-export for backward compatibility
KOHONEN_SIMPLE_MASK: list[dict[str, object]] = MASK_SIMPLE


class KohonenExperiment(Experimento):
    """Mapa autoorganizado de Kohonen con competencia lateral 2D."""

    def __init__(self) -> None:
        super().__init__()
        self._config: dict[str, Any] = {}
        self.red_tensor = None

    def setup(self, config: dict[str, Any]) -> None:
        """Configura el experimento: grilla 2D con máscara kohonen_simple."""
        self._config = config
        self.width = config.get("width", 50)
        self.height = config.get("height", 50)
        self.generation = 0

        constructor = Constructor()

        self.red, self.regiones = constructor.crear_grilla(
            width=self.width,
            height=self.height,
            filas_entrada=[],
            filas_salida=[],
            umbral=0.0,
        )

        constructor.aplicar_mascara_2d(
            self.red, self.width, self.height, KOHONEN_SIMPLE_MASK
        )

        for neurona in self.red.neuronas:
            neurona.activar_external(random.random())

        self.red_tensor = ConstructorTensor.compilar(self.red)

    def click(self, x: int, y: int) -> None:
        """Toggle: si valor < 0.5 -> activar (1.0), si >= 0.5 -> desactivar (0.0)."""
        idx = y * self.width + x
        if 0 <= idx < self.red_tensor.n_real:
            current = self.red_tensor.valores[idx].item()
            self.red_tensor.set_valor(idx, 0.0 if current >= 0.5 else 1.0)

    def step(self) -> dict[str, Any]:
        """Un step procesa toda la red de golpe. Kohonen nunca termina."""
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
        """N steps en una sola operación tensorial (sin frames intermedios)."""
        self.red_tensor.procesar_n(count)
        self.generation += count
        return {
            "type": "frame",
            "generation": self.generation,
            "grid": self.get_frame(),
            "stats": self.get_stats(),
        }

    def get_frame(self) -> list[list[float]]:
        """Retorna la grilla actual como matriz de valores."""
        if self.red_tensor:
            return self.red_tensor.get_grid(self.width, self.height)
        return super().get_frame()

    def get_tension_frame(self) -> list[list[float]] | None:
        """Retorna la grilla de tensiones superficiales."""
        if self.red_tensor:
            return self.red_tensor.get_tension_grid(self.width, self.height)
        return None

    def reset(self) -> None:
        """Reinicia el experimento con nuevos valores aleatorios."""
        self.setup(self._config)

    def is_complete(self) -> bool:
        """Kohonen nunca termina."""
        return False

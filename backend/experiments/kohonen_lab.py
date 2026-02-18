"""Experimento Kohonen Lab — laboratorio de conexionados con máscara configurable.

Permite elegir entre múltiples presets de conexionado (sombrero mexicano)
y ajustar el balance excitación/inhibición. Soporta reconexión en caliente:
cambiar máscara y balance sin perder el estado de las neuronas.
"""

from __future__ import annotations

import random
from typing import Any

from core.constructor import Constructor
from core.constructor_tensor import ConstructorTensor
from core.masks import get_mask
from .base import Experimento


class KohonenLabExperiment(Experimento):
    """Laboratorio de conexionados Kohonen con máscara configurable."""

    def __init__(self) -> None:
        super().__init__()
        self._config: dict[str, Any] = {}
        self.red_tensor = None

    def setup(self, config: dict[str, Any]) -> None:
        """Configura grilla 2D con máscara, balance e inicialización elegidos.

        Config keys:
            width (int): Ancho de la grilla (default 30).
            height (int): Alto de la grilla (default 30).
            mask (str): ID del preset de máscara (default "simple").
            balance (float | None): Balance excitación/inhibición (default None).
                0.0 = sin cambio, >0 reduce inhibición, <0 reduce excitación.
            init (str): Modo de inicialización: "random", "all_on", "all_off"
                (default "random").
        """
        self._config = config
        self.width = config.get("width", 30)
        self.height = config.get("height", 30)
        self.generation = 0

        mask_id: str = config.get("mask", "simple")
        balance = config.get("balance", None)
        init_mode: str = config.get("init", "random")
        mask = get_mask(mask_id)

        constructor = Constructor()

        self.red, self.regiones = constructor.crear_grilla(
            width=self.width,
            height=self.height,
            filas_entrada=[],
            filas_salida=[],
            umbral=0.0,
        )

        constructor.aplicar_mascara_2d(
            self.red, self.width, self.height, mask
        )

        if balance is not None:
            constructor.balancear_pesos(
                list(self.red.neuronas), target=balance
            )

        for neurona in self.red.neuronas:
            if init_mode == "all_on":
                neurona.activar_external(1.0)
            elif init_mode == "all_off":
                neurona.activar_external(0.0)
            else:
                neurona.activar_external(random.random())

        self.red_tensor = ConstructorTensor.compilar(self.red)

    def reconnect(self, config: dict[str, Any]) -> None:
        """Cambia máscara y/o balance preservando el estado de las neuronas.

        Rebuild connectivity from scratch but restore the current tensor values.
        """
        if self.red_tensor is None:
            return

        saved_values = self.red_tensor.valores[: self.red_tensor.n_real].clone()

        mask_id = config.get("mask", self._config.get("mask", "simple"))
        balance = config.get("balance", self._config.get("balance", None))
        self._config["mask"] = mask_id
        self._config["balance"] = balance

        mask = get_mask(mask_id)
        constructor = Constructor()

        self.red, self.regiones = constructor.crear_grilla(
            width=self.width,
            height=self.height,
            filas_entrada=[],
            filas_salida=[],
            umbral=0.0,
        )

        constructor.aplicar_mascara_2d(
            self.red, self.width, self.height, mask
        )

        if balance is not None:
            constructor.balancear_pesos(
                list(self.red.neuronas), target=balance
            )

        for i, neurona in enumerate(self.red.neuronas):
            neurona.activar_external(saved_values[i].item())

        self.red_tensor = ConstructorTensor.compilar(self.red)

    def click(self, x: int, y: int) -> None:
        """Toggle: si valor < 0.5 -> activar (1.0), si >= 0.5 -> desactivar (0.0)."""
        idx = y * self.width + x
        if 0 <= idx < self.red_tensor.n_real:
            current = self.red_tensor.valores[idx].item()
            self.red_tensor.set_valor(idx, 0.0 if current >= 0.5 else 1.0)

    def step(self) -> dict[str, Any]:
        """Un step procesa toda la red de golpe."""
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
        """N steps en una sola operación tensorial."""
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

    def reset(self) -> None:
        """Reinicia el experimento con la misma configuración."""
        self.setup(self._config)

    def is_complete(self) -> bool:
        """Kohonen Lab nunca termina."""
        return False

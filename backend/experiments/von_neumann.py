"""Experimento Von Neumann — autómata celular elemental (1D, reglas de Wolfram).

Usa Constructor para armar:
- Red de width × height neuronas
- Region "entrada" = fila inferior (NeuronaEntrada)
- Region "salida" = fila superior
- Region "interna" = filas intermedias

Constructor conecta cada neurona interna/salida a 3 de la fila inferior.
Constructor configura dendritas según tabla de la regla.
Procesamiento fila por fila (bottom-up), 1 fila por step.
"""

from __future__ import annotations

from typing import Any

from core.constructor import Constructor
from core.neurona import NeuronaEntrada
from .base import Experimento


class VonNeumannExperiment(Experimento):
    """Autómata celular elemental implementado con neuronas."""

    def __init__(self) -> None:
        super().__init__()
        self.rule: int = 111
        self._current_row: int = 0  # Próxima fila a procesar (bottom-up)
        self._config: dict[str, Any] = {}

    def setup(self, config: dict[str, Any]) -> None:
        """Configura el experimento con la regla y dimensiones dadas."""
        self._config = config
        self.width = config.get("width", 50)
        self.height = config.get("height", 50)
        self.rule = config.get("rule", 111)
        self.generation = 0

        constructor = Constructor()

        # Crear grilla: fila inferior = entrada, fila superior = salida
        fila_entrada = self.height - 1
        fila_salida = 0

        # Umbral 0.99: con sinapsis binarias (peso 0 o 1) y entradas binarias,
        # un match perfecto (3/3) da avg=1.0, parcial (2/3) da avg=0.667.
        # Umbral 0.99 asegura que solo matches perfectos activan la neurona.
        self.red, self.regiones = constructor.crear_grilla(
            width=self.width,
            height=self.height,
            filas_entrada=[fila_entrada],
            filas_salida=[fila_salida],
            umbral=0.99,
        )

        # Configurar dendritas según la regla de Wolfram para cada fila
        # excepto la fila de entrada (que no tiene dendritas)
        for fila in range(self.height - 2, -1, -1):
            constructor.aplicar_regla_wolfram(
                red=self.red,
                regla=self.rule,
                fila_destino=fila,
                width=self.width,
            )

        # La próxima fila a procesar es la inmediatamente arriba de la entrada
        self._current_row = self.height - 2

    def click(self, x: int, y: int) -> None:
        """Activa una neurona en la fila de entrada."""
        key = Constructor.key_by_coord(x, y)
        try:
            neurona = self.red.get_neurona(key)
            if isinstance(neurona, NeuronaEntrada):
                neurona.activar_external(1.0)
        except KeyError:
            pass

    def step(self) -> dict[str, Any]:
        """Procesa la siguiente fila (bottom-up).

        En lugar de procesar toda la red de golpe, procesamos fila por fila
        para simular la propagación temporal del autómata.
        """
        if self._current_row < 0:
            return {"type": "status", "state": "complete"}

        # Procesar solo las neuronas de la fila actual
        for x in range(self.width):
            key = Constructor.key_by_coord(x, self._current_row)
            try:
                neurona = self.red.get_neurona(key)
                neurona.procesar()
            except KeyError:
                pass

        # Activar solo las neuronas de la fila actual
        for x in range(self.width):
            key = Constructor.key_by_coord(x, self._current_row)
            try:
                neurona = self.red.get_neurona(key)
                neurona.activar()
            except KeyError:
                pass

        self._current_row -= 1
        self.generation += 1

        frame = self.get_frame()
        stats = self.get_stats()
        stats["processed_rows"] = self.generation

        return {
            "type": "frame",
            "generation": self.generation,
            "grid": frame,
            "stats": stats,
        }

    def reset(self) -> None:
        """Reinicia el experimento con la misma configuración."""
        self.setup(self._config)

    def is_complete(self) -> bool:
        """Retorna True si ya se procesaron todas las filas."""
        return self._current_row < 0

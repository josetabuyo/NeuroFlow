"""Red — contenedor tonto de neuronas.

Solo contiene neuronas. Solo las procesa a todas.
NO sabe qué es input ni output. NO tiene regiones.
"""

from __future__ import annotations

from .neurona import Neurona


class Red:
    """Contenedor de neuronas — solo las itera y procesa."""

    __slots__ = ("neuronas", "_neuronas_dict")

    def __init__(self, neuronas: list[Neurona]) -> None:
        self.neuronas: list[Neurona] = neuronas
        self._neuronas_dict: dict[str, Neurona] = {n.id: n for n in neuronas}

    def procesar(self) -> None:
        """Procesa TODAS las neuronas en dos fases:

        1. Fase de procesamiento: cada neurona evalúa sus dendritas
        2. Fase de activación: cada neurona aplica su umbral

        NeuronaEntrada simplemente no hace nada en ambas fases.
        """
        for neurona in self.neuronas:
            neurona.procesar()
        for neurona in self.neuronas:
            neurona.activar()

    def get_neurona(self, id: str) -> Neurona:
        """Retorna una neurona por su ID."""
        return self._neuronas_dict[id]

    def get_grid(self, width: int, height: int) -> list[list[float]]:
        """Retorna una matriz de valores con IDs en formato x{col}y{row}."""
        grid: list[list[float]] = []
        for row in range(height):
            fila: list[float] = []
            for col in range(width):
                key = f"x{col}y{row}"
                neurona = self._neuronas_dict.get(key)
                fila.append(neurona.valor if neurona else 0.0)
            grid.append(fila)
        return grid

    def __len__(self) -> int:
        return len(self.neuronas)

    def __repr__(self) -> str:
        return f"Red(neuronas={len(self.neuronas)})"

"""Red — contenedor de neuronas y topología.

Almacena neuronas y su conectividad (dendritas, sinapsis).
El procesamiento real se hace en RedTensor (paralelo).
"""

from __future__ import annotations

from .neurona import Neurona


class Red:
    """Contenedor de neuronas — estructura de datos para construcción."""

    __slots__ = ("neuronas", "_neuronas_dict")

    def __init__(self, neuronas: list[Neurona]) -> None:
        self.neuronas: list[Neurona] = neuronas
        self._neuronas_dict: dict[str, Neurona] = {n.id: n for n in neuronas}

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

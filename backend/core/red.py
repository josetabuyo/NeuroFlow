"""Red — neuron container and topology.

Stores neurons and their connectivity (dendrites, synapses).
The actual processing is done in RedTensor (parallel).
"""

from __future__ import annotations

from .neurona import Neurona


class Red:
    """Neuron container — data structure for construction."""

    __slots__ = ("neuronas", "_neuronas_dict")

    def __init__(self, neuronas: list[Neurona]) -> None:
        self.neuronas: list[Neurona] = neuronas
        self._neuronas_dict: dict[str, Neurona] = {n.id: n for n in neuronas}

    def get_neurona(self, id: str) -> Neurona:
        """Return a neuron by its ID."""
        return self._neuronas_dict[id]

    def get_grid(self, width: int, height: int) -> list[list[float]]:
        """Return a value matrix with IDs in x{col}y{row} format."""
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

"""Region — neuron grouping by reference.

Named group of neurons. References only, not owner.
The Network does not know regions exist.
Useful for the Constructor and the Experiment.
"""

from __future__ import annotations

from .neurona import Neurona


class Region:
    """Named group of neuron references."""

    __slots__ = ("nombre", "neuronas")

    def __init__(self, nombre: str) -> None:
        self.nombre = nombre
        self.neuronas: dict[str, Neurona] = {}

    def agregar(self, neurona: Neurona) -> None:
        """Add a neuron to the region (by reference)."""
        self.neuronas[neurona.id] = neurona

    def ids(self) -> list[str]:
        """Return the list of neuron IDs in the region."""
        return list(self.neuronas.keys())

    def valores(self) -> list[float]:
        """Return the list of neuron values in the region."""
        return [n.valor for n in self.neuronas.values()]

    def get_neurona(self, id: str) -> Neurona:
        """Return a neuron by its ID."""
        return self.neuronas[id]

    def __len__(self) -> int:
        return len(self.neuronas)

    def __repr__(self) -> str:
        return f"Region(nombre={self.nombre}, neuronas={len(self.neuronas)})"

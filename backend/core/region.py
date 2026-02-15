"""Region — agrupación de neuronas por referencia.

Grupo nombrado de neuronas. Solo referencias, no dueña.
La Red no sabe que existen regiones.
Útil para el Constructor y el Experimento.
"""

from __future__ import annotations

from .neurona import Neurona


class Region:
    """Grupo nombrado de referencias a neuronas."""

    __slots__ = ("nombre", "neuronas")

    def __init__(self, nombre: str) -> None:
        self.nombre = nombre
        self.neuronas: dict[str, Neurona] = {}

    def agregar(self, neurona: Neurona) -> None:
        """Agrega una neurona a la región (por referencia)."""
        self.neuronas[neurona.id] = neurona

    def ids(self) -> list[str]:
        """Retorna la lista de IDs de neuronas en la región."""
        return list(self.neuronas.keys())

    def valores(self) -> list[float]:
        """Retorna la lista de valores de neuronas en la región."""
        return [n.valor for n in self.neuronas.values()]

    def get_neurona(self, id: str) -> Neurona:
        """Retorna una neurona por su ID."""
        return self.neuronas[id]

    def __len__(self) -> int:
        return len(self.neuronas)

    def __repr__(self) -> str:
        return f"Region(nombre={self.nombre}, neuronas={len(self.neuronas)})"

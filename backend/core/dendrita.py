"""Dendrita — rama de entrada con múltiples sinapsis.

Lógica: valor = avg(sinapsis.procesar()) * peso_dendrita
Fuzzy AND: todas las sinapsis deben matchear para valor alto.
peso ∈ [-1, 1] — puede ser negativo (inhibición).
"""

from __future__ import annotations

from .sinapsis import Sinapsis


class Dendrita:
    """Rama dendrítica: agrupa sinapsis y las promedia."""

    __slots__ = ("sinapsis", "peso", "valor")

    def __init__(self, sinapsis: list[Sinapsis], peso: float) -> None:
        if peso < -1.0 or peso > 1.0:
            raise ValueError(f"Peso de dendrita debe estar en [-1, 1], recibido: {peso}")
        self.sinapsis = sinapsis
        self.peso = peso
        self.valor: float = 0.0

    def procesar(self) -> float:
        """Calcula el valor: promedio de sinapsis * peso de la dendrita."""
        if not self.sinapsis:
            self.valor = 0.0
            return self.valor

        suma = sum(s.procesar() for s in self.sinapsis)
        promedio = suma / len(self.sinapsis)
        self.valor = promedio * self.peso
        return self.valor

    def __repr__(self) -> str:
        return f"Dendrita(peso={self.peso:.3f}, sinapsis={len(self.sinapsis)})"

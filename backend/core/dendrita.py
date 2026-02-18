"""Dendrita — rama de entrada con múltiples sinapsis.

Almacena sinapsis y peso dendrítico. El procesamiento real
(promedio de sinapsis × peso) se hace en RedTensor.
"""

from __future__ import annotations

from .sinapsis import Sinapsis


class Dendrita:
    """Rama dendrítica: agrupa sinapsis y su peso."""

    __slots__ = ("sinapsis", "peso")

    def __init__(self, sinapsis: list[Sinapsis], peso: float) -> None:
        if peso < -1.0 or peso > 1.0:
            raise ValueError(f"Peso de dendrita debe estar en [-1, 1], recibido: {peso}")
        self.sinapsis = sinapsis
        self.peso = peso

    def __repr__(self) -> str:
        return f"Dendrita(peso={self.peso:.3f}, sinapsis={len(self.sinapsis)})"

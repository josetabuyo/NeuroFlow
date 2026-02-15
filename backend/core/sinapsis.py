"""Sinapsis — conexión sináptica pesada.

Lógica: valor = 1 - |peso - neurona_entrante.valor|

peso ∈ [0, 1] — siempre positivo.
  peso ≈ 1 reconoce entrada = 1
  peso ≈ 0 reconoce entrada = 0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .neurona import Neurona


class Sinapsis:
    """Conexión sináptica entre una neurona entrante y una dendrita."""

    __slots__ = ("neurona_entrante", "peso", "valor")

    def __init__(self, neurona_entrante: Neurona, peso: float) -> None:
        if peso < 0.0 or peso > 1.0:
            raise ValueError(f"Peso de sinapsis debe estar en [0, 1], recibido: {peso}")
        self.neurona_entrante = neurona_entrante
        self.peso = peso
        self.valor: float = 0.0

    def procesar(self) -> float:
        """Calcula el valor de la sinapsis: 1 - |peso - entrada|."""
        self.valor = 1.0 - abs(self.peso - self.neurona_entrante.valor)
        return self.valor

    def __repr__(self) -> str:
        return f"Sinapsis(peso={self.peso:.3f}, src={self.neurona_entrante.id})"

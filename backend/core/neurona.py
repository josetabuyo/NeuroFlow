"""Neurona y NeuronaEntrada — nodos del grafo neuronal.

Almacenan topología (dendritas, sinapsis) y valor de activación.
El procesamiento real se hace en RedTensor (paralelo, vectorizado).
"""

from __future__ import annotations

from .dendrita import Dendrita


class Neurona:
    """Nodo neuronal: almacena dendritas, umbral y valor de activación."""

    __slots__ = ("id", "valor", "dendritas", "umbral")

    def __init__(
        self,
        id: str,
        dendritas: list[Dendrita] | None = None,
        umbral: float = 0.0,
    ) -> None:
        self.id = id
        self.valor: float = 0.0
        self.dendritas: list[Dendrita] = dendritas if dendritas is not None else []
        self.umbral = umbral

    def activar_external(self, valor: float) -> None:
        """Setea valor directamente desde el exterior (click, inicialización)."""
        self.valor = valor

    def __repr__(self) -> str:
        return f"Neurona(id={self.id}, valor={self.valor})"


class NeuronaEntrada(Neurona):
    """Neurona de entrada: sin dendritas, valor seteado externamente.

    RedTensor la identifica y preserva su valor durante el procesamiento.
    """

    def __init__(self, id: str) -> None:
        super().__init__(id=id, dendritas=[], umbral=0.0)

    def __repr__(self) -> str:
        return f"NeuronaEntrada(id={self.id}, valor={self.valor})"

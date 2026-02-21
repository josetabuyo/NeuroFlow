"""Neurona and NeuronaEntrada — neural graph nodes.

Store topology (dendrites, synapses) and activation value.
The actual processing is done in RedTensor (parallel, vectorized).
"""

from __future__ import annotations

from .dendrita import Dendrita


class Neurona:
    """Neural node: stores dendrites, threshold, and activation value."""

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
        """Set value directly from external source (click, initialization)."""
        self.valor = valor

    def __repr__(self) -> str:
        return f"Neurona(id={self.id}, valor={self.valor})"


class NeuronaEntrada(Neurona):
    """Input neuron: no dendrites, value set externally.

    RedTensor identifies it and preserves its value during processing.
    """

    def __init__(self, id: str) -> None:
        super().__init__(id=id, dendritas=[], umbral=0.0)

    def __repr__(self) -> str:
        return f"NeuronaEntrada(id={self.id}, valor={self.valor})"

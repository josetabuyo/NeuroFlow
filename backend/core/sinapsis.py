"""Sinapsis — weighted synaptic connection.

Stores source neuron and weight. The actual processing
(1 - |weight - input|) is done in RedTensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .neurona import Neurona


class Sinapsis:
    """Synaptic connection between an incoming neuron and a dendrite."""

    __slots__ = ("neurona_entrante", "peso")

    def __init__(self, neurona_entrante: Neurona, peso: float) -> None:
        if peso < 0.0 or peso > 1.0:
            raise ValueError(f"Synapse weight must be in [0, 1], got: {peso}")
        self.neurona_entrante = neurona_entrante
        self.peso = peso

    def __repr__(self) -> str:
        return f"Sinapsis(peso={self.peso:.3f}, src={self.neurona_entrante.id})"

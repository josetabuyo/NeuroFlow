"""Dendrita — input branch with multiple synapses.

Stores synapses and dendritic weight. The actual processing
(average of synapses × weight) is done in RedTensor.
"""

from __future__ import annotations

from .sinapsis import Sinapsis


class Dendrita:
    """Dendritic branch: groups synapses and their weight."""

    __slots__ = ("sinapsis", "peso")

    def __init__(self, sinapsis: list[Sinapsis], peso: float) -> None:
        if peso < -1.0 or peso > 1.0:
            raise ValueError(f"Dendrite weight must be in [-1, 1], got: {peso}")
        self.sinapsis = sinapsis
        self.peso = peso

    def __repr__(self) -> str:
        return f"Dendrita(peso={self.peso:.3f}, sinapsis={len(self.sinapsis)})"

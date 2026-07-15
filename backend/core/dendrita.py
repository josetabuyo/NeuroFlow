"""Dendrita — input branch with multiple synapses.

Stores synapses and dendritic weight. The actual processing
(average of synapses × weight) is done in RedTensor.
"""

from __future__ import annotations

from .sinapsis import Sinapsis


class Dendrita:
    """Dendritic branch: groups synapses and their weight.

    `grupo_id` identifies which declared wiring block produced this dendrite
    (e.g. one `deamon.groups[]` entry, or one nerve/full connection) — dendrites
    sharing a `grupo_id` are the same conceptual group and get averaged
    together before group_avg splits groups by sign. Empty string means "not
    part of a declared multi-dendrite group" (e.g. Wolfram row wiring) — such
    dendrites are always their own singleton group, never merged with others.
    """

    __slots__ = ("sinapsis", "peso", "exclude_from_delta_weight", "grupo_id")

    def __init__(
        self,
        sinapsis: list[Sinapsis],
        peso: float,
        exclude_from_delta_weight: bool = False,
        grupo_id: str = "",
    ) -> None:
        if peso < -1.0 or peso > 1.0:
            raise ValueError(f"Dendrite weight must be in [-1, 1], got: {peso}")
        self.sinapsis = sinapsis
        self.peso = peso
        self.exclude_from_delta_weight = exclude_from_delta_weight
        self.grupo_id = grupo_id

    def __repr__(self) -> str:
        return f"Dendrita(peso={self.peso:.3f}, sinapsis={len(self.sinapsis)})"

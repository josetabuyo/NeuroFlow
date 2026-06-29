"""Nerve connection builder — spatial bundles of cross-region axons and synapses.

A nerve is a single circle placed on a host region. Within the circle,
tissue neurons receive connections with probability following a radial
gradient (center = 1.0, edge ≈ 0.0).
"""
from __future__ import annotations

import math
import random as _random_mod
from dataclasses import dataclass
from typing import Any


@dataclass
class NerveCircle:
    cx: int
    cy: int
    radius: int


def _place_random(
    width: int, height: int, radius: int, rng: _random_mod.Random
) -> list[NerveCircle]:
    cx = rng.randint(radius, max(radius, width - radius - 1))
    cy = rng.randint(radius, max(radius, height - radius - 1))
    return [NerveCircle(cx, cy, radius)]


def place_nerve_circles(
    nerve_cfg: dict[str, Any],
    region_width: int,
    region_height: int,
    rng: _random_mod.Random | None = None,
) -> list[NerveCircle]:
    """Place a single nerve circle according to the nerve config.

    insertion: {"x": int, "y": int}  — exact position
    insertion: "random"              — random valid position (default)
    """
    if rng is None:
        rng = _random_mod.Random()
    insertion = nerve_cfg.get("insertion", "random")
    radius = int(nerve_cfg.get("radius", 6))
    if isinstance(insertion, dict):
        cx = int(insertion.get("x", radius))
        cy = int(insertion.get("y", radius))
        return [NerveCircle(cx, cy, radius)]
    return _place_random(region_width, region_height, radius, rng)


def circle_cells_with_weights(
    circle: NerveCircle, region_width: int, region_height: int
) -> list[tuple[int, int, float]]:
    """Return (x, y, gradient_weight) for all in-bounds cells inside the circle.

    gradient_weight = 1.0 - distance/radius  (linear; center=1.0, edge≈0.0)
    """
    cells: list[tuple[int, int, float]] = []
    r = circle.radius
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= r:
                x = circle.cx + dx
                y = circle.cy + dy
                if 0 <= x < region_width and 0 <= y < region_height:
                    weight = max(0.0, 1.0 - dist / r) if r > 0 else 1.0
                    cells.append((x, y, weight))
    return cells

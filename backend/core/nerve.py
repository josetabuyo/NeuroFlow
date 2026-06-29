"""Nerve connection builder — spatial bundles of cross-region axons and synapses.

A nerve is a set of circles placed on a host region. Within each circle,
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


def _circles_overlap(a: NerveCircle, b: NerveCircle) -> bool:
    dist = math.sqrt((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2)
    return dist < 2 * a.radius - 0.5  # small tolerance


def _in_bounds(cx: int, cy: int, radius: int, width: int, height: int) -> bool:
    return radius <= cx < width - radius and radius <= cy < height - radius


def _place_random(
    width: int, height: int, count: int, radius: int, rng: _random_mod.Random
) -> list[NerveCircle]:
    circles: list[NerveCircle] = []
    max_attempts = 200
    attempts = 0
    while len(circles) < count and attempts < max_attempts:
        cx = rng.randint(radius, max(radius, width - radius - 1))
        cy = rng.randint(radius, max(radius, height - radius - 1))
        candidate = NerveCircle(cx, cy, radius)
        if not any(_circles_overlap(candidate, c) for c in circles):
            circles.append(candidate)
        attempts += 1
    return circles


def _place_line(
    width: int, height: int, count: int, radius: int, position: dict[str, int]
) -> list[NerveCircle]:
    circles: list[NerveCircle] = []
    start_x = int(position.get("x", radius))
    start_y = int(position.get("y", radius))
    for i in range(count):
        cx = start_x + i * 2 * radius
        cy = start_y
        if _in_bounds(cx, cy, radius, width, height):
            circles.append(NerveCircle(cx, cy, radius))
    return circles


def _place_random_glue(
    width: int, height: int, count: int, radius: int, rng: _random_mod.Random
) -> list[NerveCircle]:
    circles: list[NerveCircle] = []
    if count == 0:
        return circles
    cx = rng.randint(radius, max(radius, width - radius - 1))
    cy = rng.randint(radius, max(radius, height - radius - 1))
    circles.append(NerveCircle(cx, cy, radius))
    max_attempts = 300
    attempts = 0
    while len(circles) < count and attempts < max_attempts:
        base = rng.choice(circles)
        angle = rng.uniform(0.0, 2 * math.pi)
        new_cx = round(base.cx + 2 * radius * math.cos(angle))
        new_cy = round(base.cy + 2 * radius * math.sin(angle))
        if not _in_bounds(new_cx, new_cy, radius, width, height):
            attempts += 1
            continue
        candidate = NerveCircle(new_cx, new_cy, radius)
        if not any(_circles_overlap(candidate, c) for c in circles):
            circles.append(candidate)
        attempts += 1
    return circles


def place_nerve_circles(
    nerve_cfg: dict[str, Any],
    region_width: int,
    region_height: int,
    rng: _random_mod.Random | None = None,
) -> list[NerveCircle]:
    """Place nerve circles on a region according to the nerve config."""
    if rng is None:
        rng = _random_mod.Random()
    insertion = nerve_cfg.get("insertion", "random")
    count = int(nerve_cfg.get("count", 1))
    radius = int(nerve_cfg.get("radius", 6))
    position = nerve_cfg.get("position", {"x": radius, "y": radius})
    if insertion == "line":
        return _place_line(region_width, region_height, count, radius, position)
    if insertion == "random_glue":
        return _place_random_glue(region_width, region_height, count, radius, rng)
    return _place_random(region_width, region_height, count, radius, rng)


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

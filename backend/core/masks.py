"""Mask presets for Deamons Lab experiments.

Each mask is a list of dendrite definitions:
    [{"peso_dendrita": float, "offsets": [(dx, dy), ...]}, ...]

Offsets are relative to the target neuron: (dx, dy) where dx=column, dy=row.
Positive dx = right, positive dy = down.
"""

from __future__ import annotations

import math
import random as _random_mod
from typing import Any


MaskDef = list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers for generating offsets
# ---------------------------------------------------------------------------

def _moore(radius: int) -> list[tuple[int, int]]:
    """Moore neighborhood: cells within Chebyshev distance <= radius, excluding center."""
    return [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if not (dx == 0 and dy == 0)
    ]


def _von_neumann(radius: int) -> list[tuple[int, int]]:
    """Von Neumann neighborhood: cells with Manhattan distance <= radius, excluding center."""
    return [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if 0 < abs(dx) + abs(dy) <= radius
    ]


def _ring(r_inner: int, r_outer: int) -> list[tuple[int, int]]:
    """Annular ring: cells with Chebyshev distance in [r_inner, r_outer]."""
    return [
        (dx, dy)
        for dx in range(-r_outer, r_outer + 1)
        for dy in range(-r_outer, r_outer + 1)
        if r_inner <= max(abs(dx), abs(dy)) <= r_outer
    ]


def _sector_of(dx: int, dy: int, n_sectors: int) -> int:
    """Assign an offset to a sector (0..n_sectors-1), clockwise from East."""
    angle = math.atan2(-dy, dx)
    width = 2 * math.pi / n_sectors
    return int(((angle + width / 2) % (2 * math.pi)) / width) % n_sectors


def _partition(offsets: list[tuple[int, int]], n_sectors: int) -> list[list[tuple[int, int]]]:
    """Partition offsets into n_sectors directional groups."""
    sectors: list[list[tuple[int, int]]] = [[] for _ in range(n_sectors)]
    for dx, dy in offsets:
        sectors[_sector_of(dx, dy, n_sectors)].append((dx, dy))
    return [s for s in sectors if s]


def _sparse_ring(r_inner: int, r_outer: int, step: int = 2) -> list[tuple[int, int]]:
    """Sparse annular ring: keeps only cells where (dx+dy) % step == 0.

    step=2 gives ~50% density (checkerboard), step=3 gives ~33%, etc.
    """
    return [
        (dx, dy)
        for dx in range(-r_outer, r_outer + 1)
        for dy in range(-r_outer, r_outer + 1)
        if r_inner <= max(abs(dx), abs(dy)) <= r_outer
        and (dx + dy) % step == 0
    ]


def _random_sparse(
    offsets: list[tuple[int, int]],
    density: float,
    seed: int = 0,
) -> list[tuple[int, int]]:
    """Randomly subsample offsets keeping ~density fraction (deterministic via seed)."""
    rng = _random_mod.Random(seed)
    return [o for o in offsets if rng.random() < density]


def _make_inhibitory(
    offsets: list[tuple[int, int]],
    peso: float,
    n_sectors: int,
) -> list[dict[str, Any]]:
    """Create n_sectors inhibitory dendrites from offsets, one per sector."""
    if n_sectors <= 1:
        return [{"peso_dendrita": peso, "offsets": offsets}]
    sectors = _partition(offsets, n_sectors)
    return [{"peso_dendrita": peso, "offsets": sector} for sector in sectors]


def _shift(offsets: list[tuple[int, int]], sdx: int, sdy: int) -> list[tuple[int, int]]:
    """Translate every offset by (sdx, sdy)."""
    return [(dx + sdx, dy + sdy) for dx, dy in offsets]


def _ring_sq(r: int) -> list[tuple[int, int]]:
    """Single Chebyshev ring at exactly r (square shell)."""
    return _ring(r, r)


def _ring_ci(r: int) -> list[tuple[int, int]]:
    """Single Euclidean ring: cells where round(sqrt(dx²+dy²)) == r."""
    return [
        (dx, dy)
        for dx in range(-r - 1, r + 2)
        for dy in range(-r - 1, r + 2)
        if not (dx == 0 and dy == 0)
        and round(math.sqrt(dx * dx + dy * dy)) == r
    ]


# ---------------------------------------------------------------------------
# Deamon wiring compiler
# ---------------------------------------------------------------------------

DeamonWiringDef = dict[str, Any]


def _apply_noise(weights: list[float], noise: float, seed: int) -> list[float]:
    """Perturb each weight multiplicatively by ±noise, clamped to [0, 1]."""
    if noise <= 0.0:
        return weights
    rng = _random_mod.Random(seed)
    return [
        max(0.0, min(1.0, w * (1.0 + rng.uniform(-noise, noise))))
        for w in weights
    ]


def default_group_weight(group: dict[str, Any]) -> float:
    """Structural default weight for a group that doesn't set its own:
    unsectored (single dendrite) groups default to +1.0, sectored groups to
    -1.0 — the historical excitatory/inhibitory convention, driven by shape
    instead of by name."""
    return -1.0 if "sectors" in group else 1.0


def resolve_group_weight(group: dict[str, Any]) -> float:
    """A group's own ``weight`` wins; otherwise falls back to its structural
    default (see `default_group_weight`)."""
    weight = group.get("weight")
    return float(weight) if weight is not None else default_group_weight(group)


def _resolve_offsets(groups: list[dict[str, Any]], gap: int) -> list[int]:
    """Resolve each ring group's ``offset``, filling in the ones that omit it.

    A group with an explicit ``offset`` uses it as-is. A group without one
    picks up right after the previous group's outer ring plus ``gap`` empty
    rings: ``resolved = prev_resolved + len(prev["weights"]) - 1 + gap + 1``.
    ``gap`` is strictly an *intergroup* gap — it never applies before the
    first group. The first group has no "previous", so an omitted offset
    there always resolves to exactly 1 (right after the center cell, ring 0)
    regardless of ``gap``. An auto-computed offset never goes below 1 — ring
    0 is the center cell itself, never a valid ring start — even with a
    negative ``gap``.
    """
    resolved: list[int] = []
    prev_last_ring: int | None = None
    for group in groups:
        if "offset" in group:
            off = int(group["offset"])
        elif prev_last_ring is None:
            off = 1
        else:
            off = max(1, prev_last_ring + gap + 1)
        resolved.append(off)
        prev_last_ring = off + len(group["weights"]) - 1
    return resolved


def _build_group_dendrite(
    group: dict[str, Any],
    ring_fn: Any,
    grupo_id: str,
    offset: int,
) -> dict[str, Any] | None:
    """Build the single gradient dendrite for an unsectored group."""
    weight_map: dict[tuple[int, int], float] = {}
    for i, w in enumerate(group["weights"]):
        for off in ring_fn(offset + i):
            weight_map[off] = w
    if not weight_map:
        return None
    offsets = list(weight_map.keys())
    density = group.get("density", 1.0)
    if density < 1.0:
        offsets = _random_sparse(offsets, density, seed=42)
    pesos = [weight_map[off] for off in offsets]
    noise = group.get("noise")  # None = not specified; controls per-neuron scaling only
    return {
        "peso_dendrita": resolve_group_weight(group),
        "offsets": offsets,
        "pesos_sinapsis": pesos,
        "random_noise": noise if noise is not None else 0.5,
        "grupo_id": grupo_id,
    }


def compile_deamon_wiring(wiring: DeamonWiringDef) -> MaskDef:
    """Compile a deamon wiring definition into a MaskDef.

    A wiring definition is a list of dendrite ``groups``; each group is
    self-contained (its own weight, ring math, density, noise) and plays one
    of two roles inferred from the fields it carries — never from its ``id``:

    - **single dendrite** (no ``sectors``): one gradient dendrite, ring
      offsets from ``offset``/``weights``. Defaults to weight ``+1.0``
      unless the group sets its own ``weight``.
    - **partitioned** (``sectors``): split into that many angular wedges,
      one dendrite each. Defaults to weight ``-1.0`` unless the group sets
      its own ``weight``.

    A group's ``id`` (any string, e.g. ``"first_ring"``/``"second_ring"`` by
    convention) becomes every one of its dendrites' ``grupo_id`` — used by
    ``process_mode="group_avg"`` tension to vote group-by-group. Omitted →
    falls back to a positional ``"g{index}"``. The ``id`` is purely a label;
    it never determines polarity (that's always the sign of the group's
    resolved ``weight``).

    A group's ``offset`` is optional. Omitted → accumulates right after the
    previous group's outer ring, with ``wiring["gap"]`` empty rings in
    between (default 0, i.e. adjacent). ``gap`` is strictly intergroup — the
    first group, having no previous ring, always resolves an omitted offset
    to 1 (right after the center cell) regardless of ``gap``. See
    `_resolve_offsets`.

    Shapes
    ------
    ``square`` / ``crown``
        Square uses Chebyshev (square) rings, crown uses Euclidean (round)
        rings. A partitioned group's wedges cover the full ring range,
        clockwise from East. ``crown`` is the default.

        Format::

            {
                "shape": "square",
                "gap": int,  # optional, default 0 — used by groups without their own offset
                "groups": [
                    {"id": "first_ring", "offset": int, "weights": [...], "density": float},
                    {"id": "second_ring", "weights": [...],
                     "sectors": int, "density": float},
                ],
            }

    """
    groups = wiring.get("groups", [])
    shape = wiring.get("shape", "crown")
    gap = wiring.get("gap", 0)
    mask: MaskDef = []

    # ── square / crown ────────────────────────────────────────────────────
    ring_fn = _ring_sq if shape == "square" else _ring_ci
    resolved_offsets = _resolve_offsets(groups, gap)

    for group_index, group in enumerate(groups):
        grupo_id = group.get("id") or f"g{group_index}"
        offset = resolved_offsets[group_index]
        if "sectors" not in group:
            # ── Single dendrite ──────────────────────────────────────────────
            dendrite = _build_group_dendrite(group, ring_fn, grupo_id, offset)
            if dendrite:
                mask.append(dendrite)
            continue

        # ── Sectored wedges ──────────────────────────────────────────────────
        n_sectors = group["sectors"]

        weight_map: dict[tuple[int, int], float] = {}
        for i, w in enumerate(group["weights"]):
            for off in ring_fn(offset + i):
                weight_map[off] = w

        offsets = list(weight_map.keys())
        density = group.get("density", 1.0)
        if density < 1.0:
            offsets = _random_sparse(offsets, density, seed=43)

        noise = group.get("noise")  # None = not specified; controls per-neuron scaling only
        group_weight = resolve_group_weight(group)
        sectors = _partition(offsets, n_sectors)
        for sector_offsets in sectors:
            pesos = [weight_map[off] for off in sector_offsets]
            mask.append({
                "peso_dendrita": group_weight,
                "offsets": sector_offsets,
                "pesos_sinapsis": pesos,
                "random_noise": noise if noise is not None else 0.5,
                "grupo_id": grupo_id,
            })

    centroid = wiring.get("centroid")
    if centroid:
        shift = centroid.get("shift", [0, 0])
        # "twist" is a per-neuron rotational shift — depends on each neuron's
        # absolute grid position, so it can't be baked into this shared mask
        # template. It's applied later, per-neuron, in Constructor.aplicar_mascara_2d.
        if isinstance(shift, (list, tuple)):
            sdx, sdy = int(shift[0]), int(shift[1])
            if sdx != 0 or sdy != 0:
                for dendrite in mask:
                    dendrite["offsets"] = _shift(dendrite["offsets"], sdx, sdy)

    return mask


# ---------------------------------------------------------------------------
# Mask presets
# ---------------------------------------------------------------------------

# Diagnostic masks — minimal single-dendrite cases for validating activation flow
MASK_ALL_EXC: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
]

MASK_ALL_INH: MaskDef = [
    {"peso_dendrita": -1.0, "offsets": _moore(1)},
]

# simple — Mexican hat mask (Moore r=1 exc, 12 inhibitory sectors r=2-4)
MASK_SIMPLE: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    *_make_inhibitory(_ring(2, 4), -1.0, 12),
]

MASK_WIDE_HAT: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    *_make_inhibitory(_ring(2, 7), -1.0, 12),
]

MASK_NARROW_HAT: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    *_make_inhibitory(_ring(2, 3), -1.0, 12),
]

MASK_BIG_CENTER: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(2)},
    *_make_inhibitory(_ring(4, 7), -1.0, 12),
]

MASK_BIG_CENTER_WIDE_INH: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(2)},
    *_make_inhibitory(_ring(4, 10), -1.0, 12),
]

MASK_SMALL_CENTER_GAP_WIDE_INH_X2: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    *_make_inhibitory(_ring(3, 13), -1.0, 12),
]

MASK_BIG_CENTER_WIDE_INH_X2: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(2)},
    *_make_inhibitory(_ring(4, 14), -1.0, 12),
]

MASK_XL_CENTER_GAP_WIDE_INH_X2: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(5, 15), -1.0, 12),
]

MASK_DEAMON_3_EN_50: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(5, 15), -1.0, 12),
]

MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_UP: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _shift(_moore(3), 0, 1)},
    *_make_inhibitory(_shift(_ring(5, 15), 0, 1), -1.0, 12),
]

MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_DOWN: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _shift(_moore(3), 0, -1)},
    *_make_inhibitory(_shift(_ring(5, 15), 0, -1), -1.0, 12),
]

MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_LEFT: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _shift(_moore(3), 1, 0)},
    *_make_inhibitory(_shift(_ring(5, 15), 1, 0), -1.0, 12),
]

MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_RIGHT: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _shift(_moore(3), -1, 0)},
    *_make_inhibitory(_shift(_ring(5, 15), -1, 0), -1.0, 12),
]

MASK_DEAMON_1_EN_50: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_sparse_ring(5, 35, step=2), -1.0, 12),
]

MASK_DEAMON_1_5_EN_50: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(7, 17), -1.0, 12),
]

MASK_DEAMON_1_5_EN_50_G6: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(10, 20), -1.0, 12),
]

MASK_DEAMON_E3_G12_I11: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(16, 26), -1.0, 12),
]

MASK_DEAMON_E3_G12_I5: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(16, 20), -1.0, 12),
]

MASK_DEAMON_E1_G12_I1: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    *_make_inhibitory(_ring(14, 14), -1.0, 12),
]

MASK_DEAMON_E2_G12_I1: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(2)},
    *_make_inhibitory(_ring(15, 15), -1.0, 12),
]

MASK_DEAMON_E3_G12_I1: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(16, 16), -1.0, 12),
]

MASK_DEAMON_E3_G12_I3: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(16, 18), -1.0, 12),
]

MASK_DEAMON_E3_G12_I3_DE3_DI3: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _random_sparse(_moore(3), 1 / 3, seed=42)},
    *_make_inhibitory(_random_sparse(_ring(16, 18), 1 / 3, seed=43), -1.0, 12),
]

MASK_DEAMON_E3_G12_I3_DE1_DI3: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_random_sparse(_ring(16, 18), 1 / 3, seed=43), -1.0, 12),
]

MASK_DEAMON_E3_G12_I3_DE3_DI1: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _random_sparse(_moore(3), 1 / 3, seed=42)},
    *_make_inhibitory(_ring(16, 18), -1.0, 12),
]

MASK_DEAMON_E2_G3_I3_DE1_5_DI1_5: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _random_sparse(_moore(2), 2 / 3, seed=44)},
    *_make_inhibitory(_random_sparse(_ring(6, 8), 2 / 3, seed=45), -1.0, 12),
]

MASK_DEAMON_E2_G3_I3_DE1_DI1_5: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(2)},
    *_make_inhibitory(_random_sparse(_ring(6, 8), 2 / 3, seed=45), -1.0, 12),
]

MASK_DEAMON_E2_G3_I3_DE1_DI1: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(2)},
    *_make_inhibitory(_ring(6, 8), -1.0, 12),
]

MASK_DEAMON_E2_G6_I3_DE1_DI1: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(2)},
    *_make_inhibitory(_ring(9, 11), -1.0, 12),
]

MASK_DEAMON_E3_G8_I3_DE1_DI1_1: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_random_sparse(_ring(12, 14), 1 / 1.1, seed=43), -1.0, 12),
]

MASK_DEAMON_E3_G2_I12_DE1_DI1: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(6, 17), -1.0, 12),
]

# Same topology, fixed synapse weights (random_weights=False in the preset)
MASK_DEAMON_E3_G2_I12_DE1_DI1_WE1_WI1: MaskDef = MASK_DEAMON_E3_G2_I12_DE1_DI1

# Discrete Mexican-hat approximation: gradient weights per ring, square shape
_WIRING_MHAT_SQ: DeamonWiringDef = {
    "shape": "square",
    "groups": [
        {"id": "first_ring", "offset": 1, "weights": [1.0, 0.85, 0.50]},
        {
            "id": "second_ring",
            "offset": 6,
            "sectors": 12,
            "weights": [0.50, 0.85, 0.70, 0.60, 0.55, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10],
        },
    ],
    "gap": {"offset": 4, "size": 2},
}
MASK_DEAMON_E3_G2_I12_MHAT_SQ: MaskDef = compile_deamon_wiring(_WIRING_MHAT_SQ)

MASK_BIG_CENTER_SOFT_WIDE_INH: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _ring(1, 1)},
    {"peso_dendrita": 0.7, "offsets": _ring(2, 2)},
    *_make_inhibitory(_ring(4, 10), -1.0, 12),
]

MASK_CROSS_CENTER: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _von_neumann(1)},
    *_make_inhibitory(_ring(2, 4), -1.0, 4),
]

MASK_ONE_DENDRITE: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    {"peso_dendrita": -1.0, "offsets": _ring(2, 4)},
]

MASK_FINE_GRAIN: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    *_make_inhibitory(_ring(2, 4), -1.0, 16),
]

MASK_DOUBLE_RING: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    *_make_inhibitory(_ring(2, 3), -1.0, 12),
    *_make_inhibitory(_ring(5, 7), -0.5, 12),
]

MASK_SOFT_INHIBIT: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    *_make_inhibitory(_ring(2, 4), -0.5, 12),
]

MASK_STRONG_CENTER: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    *_make_inhibitory(_ring(2, 4), -1.0, 12),
]

MASK_GRADUAL_CENTER: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _ring(1, 1)},
    {"peso_dendrita": 0.6, "offsets": _ring(2, 2)},
    {"peso_dendrita": 0.3, "offsets": _ring(3, 3)},
    *_make_inhibitory(_sparse_ring(6, 11), -1.0, 12),
]

MASK_GRADUAL_BIG_INH: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _ring(1, 1)},
    {"peso_dendrita": 0.6, "offsets": _ring(2, 2)},
    {"peso_dendrita": 0.3, "offsets": _ring(3, 3)},
    *_make_inhibitory(_sparse_ring(8, 19, step=3), -1.0, 12),
]

MASK_GRADUAL_XXL_INH: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _ring(1, 1)},
    {"peso_dendrita": 0.6, "offsets": _ring(2, 2)},
    {"peso_dendrita": 0.3, "offsets": _ring(3, 3)},
    *_make_inhibitory(_sparse_ring(8, 30, step=4), -1.0, 12),
]

MASK_GRADUAL_XXL_INH_SMALL: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _ring(1, 1)},
    *_make_inhibitory(_sparse_ring(5, 30, step=4), -1.0, 12),
]

MASK_MEXICAN_HAT: MaskDef = [
    # Excitatory peak — sharp falloff
    {"peso_dendrita": 1.0, "offsets": _ring(1, 1)},
    {"peso_dendrita": 0.5, "offsets": _ring(2, 2)},
    # Inhibitory profile — strong near center, decays with distance
    *_make_inhibitory(_ring(3, 5), -1.0, 12),
    *_make_inhibitory(_sparse_ring(6, 12, step=2), -0.6, 12),
    *_make_inhibitory(_sparse_ring(13, 20, step=3), -0.25, 12),
    *_make_inhibitory(_sparse_ring(21, 30, step=5), -0.08, 12),
]


# ---------------------------------------------------------------------------
# Wolfram elementary CA rules as masks
# ---------------------------------------------------------------------------

_WOLFRAM_OFFSETS: list[tuple[int, int]] = [(-1, 1), (0, 1), (1, 1)]


def _wolfram_mask(rule: int) -> MaskDef:
    """Generate a mask for a 1D Wolfram elementary CA rule.

    Each active pattern (3 bits) becomes a dendrite with explicit synapse
    weights encoding the expected input. The fuzzy OR + high threshold (0.99)
    means the neuron fires if ANY pattern matches exactly.
    """
    dendrites: MaskDef = []
    for pattern in range(8):
        if rule & (1 << pattern):
            weights = [
                float((pattern >> 2) & 1),
                float((pattern >> 1) & 1),
                float(pattern & 1),
            ]
            dendrites.append({
                "peso_dendrita": 1.0,
                "offsets": _WOLFRAM_OFFSETS,
                "pesos_sinapsis": weights,
            })
    return dendrites


MASK_RULE_110 = _wolfram_mask(110)
MASK_RULE_30 = _wolfram_mask(30)


# ---------------------------------------------------------------------------
# Registry with metadata
# ---------------------------------------------------------------------------

MASK_PRESETS: dict[str, dict[str, Any]] = {
    "deamon_3_en_50": {
        "id": "deamon_3_en_50",
        "name": "Deamon 3 in 50",
        "description": "Moore r=3 (48 neighbors), gap r=4, corona r=5-15 (same inh area as x2).",
        "center": "Moore r=3 (48 neighbors)",
        "corona": "r=5-15, gap r=4 silence",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_3_EN_50,
    },
    "deamon_e3_g1_i11_de1_di1_move_up": {
        "id": "deamon_e3_g1_i11_de1_di1_move_up",
        "name": "Deamon (E3 G1 I11 DE1 DI1) Move Up",
        "description": "Moore r=3 (48 neighbors) shifted +1 dy, gap r=4, corona r=5-15 shifted +1 dy.",
        "center": "Moore r=3 (48 neighbors, shifted 1px down)",
        "corona": "r=5-15 shifted 1px down, gap r=4 silence",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_UP,
    },
    "deamon_e3_g1_i11_de1_di1_move_down": {
        "id": "deamon_e3_g1_i11_de1_di1_move_down",
        "name": "Deamon (E3 G1 I11 DE1 DI1) Move Down",
        "description": "Moore r=3 (48 neighbors) shifted -1 dy, gap r=4, corona r=5-15 shifted -1 dy.",
        "center": "Moore r=3 (48 neighbors, shifted 1px up)",
        "corona": "r=5-15 shifted 1px up, gap r=4 silence",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_DOWN,
    },
    "deamon_e3_g1_i11_de1_di1_move_left": {
        "id": "deamon_e3_g1_i11_de1_di1_move_left",
        "name": "Deamon (E3 G1 I11 DE1 DI1) Move Left",
        "description": "Moore r=3 (48 neighbors) shifted +1 dx, gap r=4, corona r=5-15 shifted +1 dx.",
        "center": "Moore r=3 (48 neighbors, shifted 1px right)",
        "corona": "r=5-15 shifted 1px right, gap r=4 silence",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_LEFT,
    },
    "deamon_e3_g1_i11_de1_di1_move_right": {
        "id": "deamon_e3_g1_i11_de1_di1_move_right",
        "name": "Deamon (E3 G1 I11 DE1 DI1) Move Right",
        "description": "Moore r=3 (48 neighbors) shifted -1 dx, gap r=4, corona r=5-15 shifted -1 dx.",
        "center": "Moore r=3 (48 neighbors, shifted 1px left)",
        "corona": "r=5-15 shifted 1px left, gap r=4 silence",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_RIGHT,
    },
    "deamon_1_en_50": {
        "id": "deamon_1_en_50",
        "name": "Deamon 1 in 50",
        "description": "Moore r=3 (48 neighbors), gap r=4, sparse corona r=5-35 (massive inhibition, ~1 deamon in 50x50).",
        "center": "Moore r=3 (48 neighbors)",
        "corona": "r=5-35 sparse step=2, gap r=4 silence",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_1_EN_50,
    },
    "deamon_1_5_en_50": {
        "id": "deamon_1_5_en_50",
        "name": "Deamon 1.5 in 50 (E3 G3 I11)",
        "description": "Moore r=3 (48 neighbors), gap x3 r=4-6, corona r=7-17. ~1.5 deamons in 50x50.",
        "center": "Moore r=3 (48 neighbors)",
        "corona": "r=7-17, gap r=4-6 silence (x3)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_1_5_EN_50,
    },
    "deamon_1_5_en_50_g6": {
        "id": "deamon_1_5_en_50_g6",
        "name": "Deamon 1.5 in 50 (E3 G6 I11)",
        "description": "Moore r=3 (48 neighbors), gap x6 r=4-9, corona r=10-20. ~1.5 deamons in 50x50.",
        "center": "Moore r=3 (48 neighbors)",
        "corona": "r=10-20, gap r=4-9 silence (x6)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_1_5_EN_50_G6,
    },
    "deamon_e3_g12_i11": {
        "id": "deamon_e3_g12_i11",
        "name": "Deamon (E3 G12 I11)",
        "description": "Moore r=3 (48 neighbors), gap x12 r=4-15, corona r=16-26.",
        "center": "Moore r=3 (48 neighbors)",
        "corona": "r=16-26, gap r=4-15 silence (x12)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G12_I11,
    },
    "deamon_e3_g12_i5": {
        "id": "deamon_e3_g12_i5",
        "name": "Deamon (E3 G12 I5)",
        "description": "Moore r=3 (48 neighbors), gap x12 r=4-15, corona r=16-20 (half of I11).",
        "center": "Moore r=3 (48 neighbors)",
        "corona": "r=16-20, gap r=4-15 silence (x12)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G12_I5,
    },
    "deamon_e1_g12_i1": {
        "id": "deamon_e1_g12_i1",
        "name": "Deamon (E1 G12 I1)",
        "description": "Moore r=1 (8 neighbors), gap x12 r=2-13, corona r=14 (1 ring).",
        "center": "Moore r=1 (8 neighbors)",
        "corona": "r=14, gap r=2-13 silence (x12)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E1_G12_I1,
    },
    "deamon_e2_g12_i1": {
        "id": "deamon_e2_g12_i1",
        "name": "Deamon (E2 G12 I1)",
        "description": "Moore r=2 (24 neighbors), gap x12 r=3-14, corona r=15 (1 ring).",
        "center": "Moore r=2 (24 neighbors)",
        "corona": "r=15, gap r=3-14 silence (x12)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E2_G12_I1,
    },
    "deamon_e3_g12_i1": {
        "id": "deamon_e3_g12_i1",
        "name": "Deamon (E3 G12 I1)",
        "description": "Moore r=3 (48 neighbors), gap x12 r=4-15, corona r=16 (1 ring).",
        "center": "Moore r=3 (48 neighbors)",
        "corona": "r=16, gap r=4-15 silence (x12)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G12_I1,
    },
    "deamon_e3_g12_i3": {
        "id": "deamon_e3_g12_i3",
        "name": "Deamon (E3 G12 I3)",
        "description": "Moore r=3 (48 neighbors), gap x12 r=4-15, corona r=16-18 (3 rings).",
        "center": "Moore r=3 (48 neighbors)",
        "corona": "r=16-18, gap r=4-15 silence (x12)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G12_I3,
    },
    "deamon_e3_g12_i3_de3_di3": {
        "id": "deamon_e3_g12_i3_de3_di3",
        "name": "Deamon (E3 G12 I3 DE3 DI3)",
        "description": "E3 G12 I3 with 1/3 density in exc. and inh.: ~33% random synapses in both zones.",
        "center": "Moore r=3 sparse ~33% (~16 neighbors)",
        "corona": "r=16-18 sparse ~33%, gap r=4-15 silence (x12)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G12_I3_DE3_DI3,
    },
    "deamon_e3_g12_i3_de1_di3": {
        "id": "deamon_e3_g12_i3_de1_di3",
        "name": "Deamon (E3 G12 I3 DE1 DI3)",
        "description": "E3 G12 I3 with full exc. (48 neighbors) and sparse inh. ~33%.",
        "center": "Moore r=3 (48 neighbors, full density)",
        "corona": "r=16-18 sparse ~33%, gap r=4-15 silence (x12)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G12_I3_DE1_DI3,
    },
    "deamon_e3_g12_i3_de3_di1": {
        "id": "deamon_e3_g12_i3_de3_di1",
        "name": "Deamon (E3 G12 I3 DE3 DI1)",
        "description": "E3 G12 I3 with sparse exc. ~33% and full inh. (3 rings).",
        "center": "Moore r=3 sparse ~33% (~16 neighbors)",
        "corona": "r=16-18 full, gap r=4-15 silence (x12)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G12_I3_DE3_DI1,
    },
    "deamon_e2_g3_i3_de1_5_di1_5": {
        "id": "deamon_e2_g3_i3_de1_5_di1_5",
        "name": "Deamon (E2 G3 I3 DE1.5 DI1.5)",
        "description": "Moore r=2 sparse ~67%, gap r=3-5, corona r=6-8 sparse ~67%.",
        "center": "Moore r=2 sparse ~67% (~16 neighbors)",
        "corona": "r=6-8 sparse ~67%, gap r=3-5 silence (x3)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E2_G3_I3_DE1_5_DI1_5,
    },
    "deamon_e2_g3_i3_de1_di1_5": {
        "id": "deamon_e2_g3_i3_de1_di1_5",
        "name": "Deamon (E2 G3 I3 DE1 DI1.5)",
        "description": "Moore r=2 full (24 neighbors), gap r=3-5, corona r=6-8 sparse ~67%.",
        "center": "Moore r=2 (24 neighbors, full density)",
        "corona": "r=6-8 sparse ~67%, gap r=3-5 silence (x3)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E2_G3_I3_DE1_DI1_5,
    },
    "deamon_e2_g3_i3_de1_di1": {
        "id": "deamon_e2_g3_i3_de1_di1",
        "name": "Deamon (E2 G3 I3 DE1 DI1)",
        "description": "Moore r=2 full (24 neighbors), gap r=3-5, corona r=6-8 full.",
        "center": "Moore r=2 (24 neighbors, full density)",
        "corona": "r=6-8 full, gap r=3-5 silence (x3)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E2_G3_I3_DE1_DI1,
    },
    "deamon_e2_g6_i3_de1_di1": {
        "id": "deamon_e2_g6_i3_de1_di1",
        "name": "Deamon (E2 G6 I3 DE1 DI1)",
        "description": "Moore r=2 full (24 neighbors), gap r=3-8, corona r=9-11 full.",
        "center": "Moore r=2 (24 neighbors, full density)",
        "corona": "r=9-11 full, gap r=3-8 silence (x6)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E2_G6_I3_DE1_DI1,
    },
    "deamon_e3_g8_i3_de1_di1_1": {
        "id": "deamon_e3_g8_i3_de1_di1_1",
        "name": "Deamon (E3 G8 I3 DE1 DI1.1)",
        "description": "Moore r=3 full (48 neighbors), gap r=4-11, corona r=12-14 sparse ~91%.",
        "center": "Moore r=3 (48 neighbors, full density)",
        "corona": "r=12-14 sparse ~91%, gap r=4-11 silence (x8)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G8_I3_DE1_DI1_1,
    },
    "deamon_e3_g2_i12_de1_di1": {
        "id": "deamon_e3_g2_i12_de1_di1",
        "name": "Deamon (E3 G2 I12 DE1 DI1)",
        "description": "Moore r=3 full (48 neighbors), gap r=4-5, corona r=6-17 full.",
        "center": "Moore r=3 (48 neighbors, full density)",
        "corona": "r=6-17 full, gap r=4-5 silence (x2)",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G2_I12_DE1_DI1,
    },
    "deamon_e3_g2_i12_de1_di1_we1_wi1": {
        "id": "deamon_e3_g2_i12_de1_di1_we1_wi1",
        "name": "Deamon (E3 G2 I12 DE1 DI1 WE1 WI1)",
        "description": "Moore r=3 full (48 neighbors), gap r=4-5, corona r=6-17 full. Fixed weights: we=1, wi=-1.",
        "center": "Moore r=3 (48 neighbors, full density)",
        "corona": "r=6-17 full, gap r=4-5 silence (x2)",
        "dendrites_inh": 12,
        "random_weights": False,
        "mask": MASK_DEAMON_E3_G2_I12_DE1_DI1_WE1_WI1,
    },
    "deamon_e3_g2_i12_mhat_sq": {
        "id": "deamon_e3_g2_i12_mhat_sq",
        "name": "Deamon (E3 G2 I12 MHat Sq)",
        "description": "Discrete Mexican-hat approximation, square rings. Exc r=1-3 gradient [1.0→0.85→0.50], gap r=4-5, inh r=6-16 gradient [0.50→0.85→…→0.10].",
        "center": "Square rings r=1-3, gradient weights",
        "corona": "Square rings r=6-16, gradient weights, gap r=4-5",
        "dendrites_inh": 11,
        "random_weights": False,
        "wiring": _WIRING_MHAT_SQ,
        "mask": MASK_DEAMON_E3_G2_I12_MHAT_SQ,
    },
    "all_exc": {
        "id": "all_exc",
        "name": "All Exc",
        "description": "1 exc. dendrite r=1 (8 neighbors).",
        "center": "Moore r=1 (8 neighbors)",
        "corona": "no inhibition",
        "dendrites_inh": 0,
        "random_weights": True,
        "mask": MASK_ALL_EXC,
    },
    "all_inh": {
        "id": "all_inh",
        "name": "All Inh",
        "description": "1 inh. dendrite r=1 (8 neighbors).",
        "center": "Moore r=1 (8 neighbors)",
        "corona": "no excitation",
        "dendrites_inh": 1,
        "random_weights": True,
        "mask": MASK_ALL_INH,
    },
    "simple": {
        "id": "simple",
        "name": "Kohonen Simple",
        "description": "Moore r=1, corona r=2-4, 8 inh. dendrites.",
        "center": "Moore r=1 (8 neighbors)",
        "corona": "r=2-4, 8 blocks 3x3",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_SIMPLE,
    },
    "wide_hat": {
        "id": "wide_hat",
        "name": "Wide Hat",
        "description": "Moore r=1, corona r=2-7, 8 inh. dendrites.",
        "center": "Moore r=1 (8 neighbors)",
        "corona": "r=2-7, large corona",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_WIDE_HAT,
    },
    "narrow_hat": {
        "id": "narrow_hat",
        "name": "Narrow Hat",
        "description": "Moore r=1, corona r=2-3, 8 inh. dendrites.",
        "center": "Moore r=1 (8 neighbors)",
        "corona": "r=2-3, close corona",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_NARROW_HAT,
    },
    "big_center": {
        "id": "big_center",
        "name": "Big Center",
        "description": "Moore r=2 (24 neighbors), corona r=4-7, 8 inh. dendrites.",
        "center": "Moore r=2 (24 neighbors)",
        "corona": "r=4-7, far corona",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_BIG_CENTER,
    },
    "big_center_wide_inh": {
        "id": "big_center_wide_inh",
        "name": "Big Center Wide Inh",
        "description": "Moore r=2 (24 neighbors), corona r=4-10 (2x inh area).",
        "center": "Moore r=2 (24 neighbors)",
        "corona": "r=4-10, extended corona",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_BIG_CENTER_WIDE_INH,
    },
    "small_center_gap_wide_inh_x2": {
        "id": "small_center_gap_wide_inh_x2",
        "name": "Small Center + Gap + Wide Inh x2",
        "description": "Moore r=1 (8 neighbors), gap r=2, corona r=3-13 (same inh area as x2).",
        "center": "Moore r=1 (8 neighbors)",
        "corona": "r=3-13, gap r=2 silence",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_SMALL_CENTER_GAP_WIDE_INH_X2,
    },
    "big_center_wide_inh_x2": {
        "id": "big_center_wide_inh_x2",
        "name": "Big Center Wide Inh x2",
        "description": "Moore r=2 (24 neighbors), corona r=4-14 (4x original inh area).",
        "center": "Moore r=2 (24 neighbors)",
        "corona": "r=4-14, very extended corona",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_BIG_CENTER_WIDE_INH_X2,
    },
    "xl_center_gap_wide_inh_x2": {
        "id": "xl_center_gap_wide_inh_x2",
        "name": "XL Center + Gap + Wide Inh x2",
        "description": "Moore r=3 (48 neighbors), gap r=4, corona r=5-15 (same inh area as x2).",
        "center": "Moore r=3 (48 neighbors)",
        "corona": "r=5-15, gap r=4 silence",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_XL_CENTER_GAP_WIDE_INH_X2,
    },
    "big_center_soft_wide_inh": {
        "id": "big_center_soft_wide_inh",
        "name": "Big Center Soft Wide Inh",
        "description": "Exc. r=1(1.0) r=2(0.7) soft edge, corona r=4-10.",
        "center": "r=1→1.0, r=2→0.7 (soft edge)",
        "corona": "r=4-10, extended corona",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_BIG_CENTER_SOFT_WIDE_INH,
    },
    "cross_center": {
        "id": "cross_center",
        "name": "Cross Center",
        "description": "Von Neumann r=1 (4 neighbors), corona r=2-4, 4 inh. dendrites.",
        "center": "Von Neumann r=1 (4 neighbors)",
        "corona": "r=2-4, 4 cardinal blocks",
        "dendrites_inh": 4,
        "random_weights": True,
        "mask": MASK_CROSS_CENTER,
    },
    "one_dendrite": {
        "id": "one_dendrite",
        "name": "One Dendrite",
        "description": "Moore r=1, corona r=2-4 in 1 single inh. dendrite.",
        "center": "Moore r=1 (8 neighbors)",
        "corona": "r=2-4, all in 1 dendrite",
        "dendrites_inh": 1,
        "random_weights": True,
        "mask": MASK_ONE_DENDRITE,
    },
    "fine_grain": {
        "id": "fine_grain",
        "name": "Fine Grain",
        "description": "Moore r=1, corona r=2-4, 16 inh. sectors.",
        "center": "Moore r=1 (8 neighbors)",
        "corona": "r=2-4, 16 sectors",
        "dendrites_inh": 16,
        "random_weights": True,
        "mask": MASK_FINE_GRAIN,
    },
    "double_ring": {
        "id": "double_ring",
        "name": "Double Ring",
        "description": "Moore r=1, ring r=2-3 (-1) + ring r=5-7 (-0.5).",
        "center": "Moore r=1 (8 neighbors)",
        "corona": "r=2-3 (-1) + r=5-7 (-0.5)",
        "dendrites_inh": 24,
        "random_weights": True,
        "mask": MASK_DOUBLE_RING,
    },
    "soft_inhibit": {
        "id": "soft_inhibit",
        "name": "Soft Inhibition",
        "description": "Moore r=1, corona r=2-4, inh. weight -0.5.",
        "center": "Moore r=1 (8 neighbors)",
        "corona": "r=2-4, weight -0.5",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_SOFT_INHIBIT,
    },
    "strong_center": {
        "id": "strong_center",
        "name": "Strong Center",
        "description": "Moore r=1 x2 exc. dendrites, corona r=2-4.",
        "center": "Moore r=1 (2 exc. dendrites)",
        "corona": "r=2-4, weight -1",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_STRONG_CENTER,
    },
    "gradual_center": {
        "id": "gradual_center",
        "name": "Gradual Center",
        "description": "Gradual exc. r=1(1.0) r=2(0.6) r=3(0.3), gap 2px, sparse inh. r=6-11.",
        "center": "Gradual r=1→1.0, r=2→0.6, r=3→0.3",
        "corona": "r=6-11, checkerboard sparse",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_GRADUAL_CENTER,
    },
    "gradual_big_inh": {
        "id": "gradual_big_inh",
        "name": "Gradual Center Big Inh",
        "description": "Gradual exc. r=1-3, gap 4px, sparse inh. r=8-19.",
        "center": "Gradual r=1→1.0, r=2→0.6, r=3→0.3",
        "corona": "r=8-19, sparse step=3",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_GRADUAL_BIG_INH,
    },
    "gradual_xxl_inh": {
        "id": "gradual_xxl_inh",
        "name": "Gradual Center XXL Inh",
        "description": "Gradual exc. r=1-3, gap 4px, sparse inh. r=8-30.",
        "center": "Gradual r=1→1.0, r=2→0.6, r=3→0.3",
        "corona": "r=8-30, sparse step=4",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_GRADUAL_XXL_INH,
    },
    "gradual_xxl_inh_small": {
        "id": "gradual_xxl_inh_small",
        "name": "Small Center XXL Inh",
        "description": "Exc. r=1 only, gap 3px, sparse inh. r=5-30.",
        "center": "r=1→1.0 (immediate neighbors only)",
        "corona": "r=5-30, sparse step=4",
        "dendrites_inh": 12,
        "random_weights": True,
        "mask": MASK_GRADUAL_XXL_INH_SMALL,
    },
    "mexican_hat": {
        "id": "mexican_hat",
        "name": "Mexican Hat",
        "description": "Theoretical DoG: exc. r=1-2, gradual inh. r=3-30 (weight decays with distance).",
        "center": "Gradual r=1→1.0, r=2→0.5",
        "corona": "r=3-5(-1) → r=6-12(-0.6) → r=13-20(-0.25) → r=21-30(-0.08)",
        "dendrites_inh": 48,
        "random_weights": True,
        "mask": MASK_MEXICAN_HAT,
    },
    "rule_110": {
        "id": "rule_110",
        "name": "Wolfram Rule 110",
        "description": "Elementary cellular automaton Rule 110 (Turing-complete).",
        "center": "3 neighbors bottom row",
        "corona": "no inhibition",
        "dendrites_inh": 0,
        "random_weights": False,
        "mask_type": "wolfram",
        "mask": MASK_RULE_110,
    },
    "rule_30": {
        "id": "rule_30",
        "name": "Wolfram Rule 30",
        "description": "Elementary cellular automaton Rule 30 (chaotic, pseudo-random).",
        "center": "3 neighbors bottom row",
        "corona": "no inhibition",
        "dendrites_inh": 0,
        "random_weights": False,
        "mask_type": "wolfram",
        "mask": MASK_RULE_30,
    },
}


def get_mask(mask_id: str) -> MaskDef:
    """Get a mask definition by its ID. Raises KeyError if not found."""
    return MASK_PRESETS[mask_id]["mask"]


def get_mask_type(mask_id: str) -> str:
    """Get the mask type: 'kohonen' (default) or 'wolfram'."""
    return MASK_PRESETS[mask_id].get("mask_type", "kohonen")


def get_random_weights(mask_id: str) -> bool:
    """Whether the mask uses random synapse weights (True) or fixed 1.0 (False)."""
    return MASK_PRESETS[mask_id].get("random_weights", True)


def _mask_extent(mask: MaskDef) -> int:
    """Max Chebyshev distance from center among all synapse offsets in the mask."""
    max_r = 0
    for dendrite in mask:
        for dx, dy in dendrite["offsets"]:
            max_r = max(max_r, abs(dx), abs(dy))
    return max_r


def _preview_grid_size(mask: MaskDef, margin: int = 2) -> int:
    """Grid side length that fits every synapse in `mask` plus a small margin,
    so the preview never has to clip/drop cells — the frontend scales the
    whole grid down to fit its container, so a bigger grid just renders smaller."""
    return 2 * (_mask_extent(mask) + margin) + 1


def _compute_preview_grid(
    mask: MaskDef,
    grid_width: int = 50,
    grid_height: int = 50,
    random_weights: bool = True,
) -> list[list[float | None]]:
    """Simulate the inspect view for a center neuron.

    Creates a (grid_width × grid_height) grid and places the neuron at the
    center.  For each synapse, generates a deterministic random weight (if
    random_weights=True) or uses 1.0 (if random_weights=False), and computes
    ``effective_weight = synapse_peso × dendrite_peso``, matching how
    ``inspect()`` displays real connections.  When multiple dendrites share a
    source cell, effective weights are summed and clamped to [-1, 1].
    """
    rng = _random_mod.Random(42)

    center_x = grid_width // 2
    center_y = grid_height // 2

    grid: list[list[float | None]] = [[None] * grid_width for _ in range(grid_height)]
    grid[center_y][center_x] = 999.0

    for dendrite in mask:
        peso_d: float = dendrite["peso_dendrita"]
        pesos_s = dendrite.get("pesos_sinapsis")
        noise_amp = dendrite.get("random_noise")  # None for presets, float for inline deamons
        for i, (dx, dy) in enumerate(dendrite["offsets"]):
            col = center_x + dx
            row = center_y + dy
            if 0 <= row < grid_height and 0 <= col < grid_width:
                base = pesos_s[i] if pesos_s else 1.0
                if not random_weights:
                    syn_w = base
                elif noise_amp is not None:
                    scale = rng.uniform(1.0 - noise_amp, 1.0) if noise_amp > 0 else 1.0
                    syn_w = base * scale
                else:
                    syn_w = base * rng.random() if pesos_s else rng.random()
                effective = syn_w * peso_d
                existing = grid[row][col]
                if existing is not None and existing != 999.0:
                    effective = max(-1.0, min(1.0, existing + effective))
                grid[row][col] = effective

    return grid


def _compute_mask_stats(mask: MaskDef) -> dict[str, Any]:
    """Compute static wiring stats for a mask definition.

    Returns per-neuron synapse/dendrite counts and effective radii (Chebyshev distance).
    """
    exc_synapses = 0
    inh_synapses = 0
    exc_dendrites = 0
    inh_dendrites = 0
    max_exc_radius = 0
    max_inh_radius = 0

    for dendrite in mask:
        peso: float = dendrite["peso_dendrita"]
        offsets = dendrite["offsets"]
        n = len(offsets)
        max_r = max((max(abs(dx), abs(dy)) for dx, dy in offsets), default=0)
        if peso > 0:
            exc_synapses += n
            exc_dendrites += 1
            max_exc_radius = max(max_exc_radius, max_r)
        else:
            inh_synapses += n
            inh_dendrites += 1
            max_inh_radius = max(max_inh_radius, max_r)

    return {
        "total_dendrites": exc_dendrites + inh_dendrites,
        "exc_dendrites": exc_dendrites,
        "inh_dendrites": inh_dendrites,
        "excitatory_synapses": exc_synapses,
        "inhibitory_synapses": inh_synapses,
        "ratio_exc_inh": round(exc_synapses / max(inh_synapses, 1), 3),
        "excitation_radius": max_exc_radius,
        "inhibition_radius": max_inh_radius,
    }


def _compute_dendrite_info(
    mask: MaskDef,
    grid_width: int,
    grid_height: int,
) -> list[dict[str, Any]]:
    """Return per-dendrite info for the preview: centroid, avg weight, cell list."""
    cx = grid_width // 2
    cy = grid_height // 2
    result = []
    for dendrite in mask:
        peso_d: float = dendrite["peso_dendrita"]
        pesos_s: list[float] | None = dendrite.get("pesos_sinapsis")
        valid_cells: list[list[int]] = []
        eff_weights: list[float] = []
        for i, (dx, dy) in enumerate(dendrite["offsets"]):
            col, row = cx + dx, cy + dy
            if 0 <= col < grid_width and 0 <= row < grid_height:
                valid_cells.append([col, row])
                syn = pesos_s[i] if pesos_s else 1.0
                eff_weights.append(syn * peso_d)
        if not valid_cells:
            continue
        centroid_col = sum(c for c, _ in valid_cells) / len(valid_cells)
        centroid_row = sum(r for _, r in valid_cells) / len(valid_cells)
        avg_eff = sum(eff_weights) / len(eff_weights)
        result.append({
            "centroid": [centroid_col, centroid_row],
            "avg_effective": round(avg_eff, 4),
            "cells": valid_cells,
        })
    return result


def get_mask_info(
    grid_width: int | None = None,
    grid_height: int | None = None,
) -> list[dict[str, Any]]:
    """Get metadata for all mask presets (without the mask data itself)."""
    result = []
    for preset in MASK_PRESETS.values():
        mask = preset["mask"]
        size = _preview_grid_size(mask)
        gw = grid_width or size
        gh = grid_height or size
        entry = {k: v for k, v in preset.items() if k != "mask"}
        entry["preview_grid"] = _compute_preview_grid(
            mask, gw, gh,
            random_weights=preset.get("random_weights", True),
        )
        entry["mask_stats"] = _compute_mask_stats(mask)
        entry["dendrites"] = _compute_dendrite_info(mask, gw, gh)
        result.append(entry)
    return result


def preview_deamon_wiring(
    wiring: DeamonWiringDef,
    grid_width: int | None = None,
    grid_height: int | None = None,
) -> dict[str, Any]:
    """Compute preview_grid, mask_stats and dendrites for an inline deamon wiring."""
    mask_name = wiring.get("mask")
    if mask_name and isinstance(mask_name, str) and mask_name in MASK_PRESETS:
        preset = MASK_PRESETS[mask_name]
        mask = preset["mask"]
        random_weights = preset.get("random_weights", True)
    else:
        mask = compile_deamon_wiring(wiring)
        random_weights = not wiring.get("fixed", False)
    size = _preview_grid_size(mask)
    gw = grid_width or size
    gh = grid_height or size
    return {
        "preview_grid": _compute_preview_grid(mask, gw, gh, random_weights=random_weights),
        "mask_stats": _compute_mask_stats(mask),
        "dendrites": _compute_dendrite_info(mask, gw, gh),
    }

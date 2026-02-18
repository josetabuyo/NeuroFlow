"""Mask presets for Kohonen-type experiments.

Each mask is a list of dendrite definitions:
    [{"peso_dendrita": float, "offsets": [(dx, dy), ...]}, ...]

Offsets are relative to the target neuron: (dx, dy) where dx=column, dy=row.
Positive dx = right, positive dy = down.
"""

from __future__ import annotations

import math
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

# simple — exact copy of the original KOHONEN_SIMPLE_MASK
MASK_SIMPLE: MaskDef = [
    {
        "peso_dendrita": 1.0,
        "offsets": [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ],
    },
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (2, -4), (2, -3), (2, -2),
            (3, -4), (3, -3), (3, -2),
            (4, -4), (4, -3), (4, -2),
        ],
    },
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (2, -1), (2, 0), (2, 1),
            (3, -1), (3, 0), (3, 1),
            (4, -1), (4, 0), (4, 1),
        ],
    },
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (2, 2), (2, 3), (2, 4),
            (3, 2), (3, 3), (3, 4),
            (4, 2), (4, 3), (4, 4),
        ],
    },
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (-1, 2), (-1, 3), (-1, 4),
            (0, 2),  (0, 3),  (0, 4),
            (1, 2),  (1, 3),  (1, 4),
        ],
    },
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (-4, 2), (-4, 3), (-4, 4),
            (-3, 2), (-3, 3), (-3, 4),
            (-2, 2), (-2, 3), (-2, 4),
        ],
    },
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (-4, -1), (-4, 0), (-4, 1),
            (-3, -1), (-3, 0), (-3, 1),
            (-2, -1), (-2, 0), (-2, 1),
        ],
    },
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (-4, -4), (-4, -3), (-4, -2),
            (-3, -4), (-3, -3), (-3, -2),
            (-2, -4), (-2, -3), (-2, -2),
        ],
    },
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (-1, -4), (-1, -3), (-1, -2),
            (0, -4),  (0, -3),  (0, -2),
            (1, -4),  (1, -3),  (1, -2),
        ],
    },
]

MASK_WIDE_HAT: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    *_make_inhibitory(_ring(2, 7), -1.0, 8),
]

MASK_NARROW_HAT: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    *_make_inhibitory(_ring(2, 3), -1.0, 8),
]

MASK_BIG_CENTER: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(2)},
    *_make_inhibitory(_ring(4, 7), -1.0, 8),
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
    *_make_inhibitory(_ring(2, 3), -1.0, 8),
    *_make_inhibitory(_ring(5, 7), -0.5, 8),
]

MASK_SOFT_INHIBIT: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    *_make_inhibitory(_ring(2, 4), -0.5, 8),
]

MASK_STRONG_CENTER: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    *_make_inhibitory(_ring(2, 4), -1.0, 8),
]

MASK_GRADUAL_CENTER: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _ring(1, 1)},
    {"peso_dendrita": 0.6, "offsets": _ring(2, 2)},
    {"peso_dendrita": 0.3, "offsets": _ring(3, 3)},
    *_make_inhibitory(_sparse_ring(6, 11), -1.0, 8),
]

MASK_GRADUAL_BIG_INH: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _ring(1, 1)},
    {"peso_dendrita": 0.6, "offsets": _ring(2, 2)},
    {"peso_dendrita": 0.3, "offsets": _ring(3, 3)},
    *_make_inhibitory(_sparse_ring(8, 19, step=3), -1.0, 8),
]

MASK_GRADUAL_XXL_INH: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _ring(1, 1)},
    {"peso_dendrita": 0.6, "offsets": _ring(2, 2)},
    {"peso_dendrita": 0.3, "offsets": _ring(3, 3)},
    *_make_inhibitory(_sparse_ring(8, 30, step=4), -1.0, 8),
]


# ---------------------------------------------------------------------------
# Registry with metadata
# ---------------------------------------------------------------------------

MASK_PRESETS: dict[str, dict[str, Any]] = {
    "all_exc": {
        "id": "all_exc",
        "name": "Todo Exc",
        "description": "1 dendrita exc. r=1 (8 vecinos).",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "sin inhibición",
        "dendrites_inh": 0,
        "mask": MASK_ALL_EXC,
    },
    "all_inh": {
        "id": "all_inh",
        "name": "Todo Inh",
        "description": "1 dendrita inh. r=1 (8 vecinos).",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "sin excitación",
        "dendrites_inh": 1,
        "mask": MASK_ALL_INH,
    },
    "simple": {
        "id": "simple",
        "name": "Kohonen Simple",
        "description": "Moore r=1, corona r=2-4, 8 dendritas inh.",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=2-4, 8 bloques 3x3",
        "dendrites_inh": 8,
        "mask": MASK_SIMPLE,
    },
    "wide_hat": {
        "id": "wide_hat",
        "name": "Sombrero Ancho",
        "description": "Moore r=1, corona r=2-7, 8 dendritas inh.",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=2-7, corona grande",
        "dendrites_inh": 8,
        "mask": MASK_WIDE_HAT,
    },
    "narrow_hat": {
        "id": "narrow_hat",
        "name": "Sombrero Estrecho",
        "description": "Moore r=1, corona r=2-3, 8 dendritas inh.",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=2-3, corona cercana",
        "dendrites_inh": 8,
        "mask": MASK_NARROW_HAT,
    },
    "big_center": {
        "id": "big_center",
        "name": "Centro Grande",
        "description": "Moore r=2 (24 vecinos), corona r=4-7, 8 dendritas inh.",
        "center": "Moore r=2 (24 vecinos)",
        "corona": "r=4-7, corona lejana",
        "dendrites_inh": 8,
        "mask": MASK_BIG_CENTER,
    },
    "cross_center": {
        "id": "cross_center",
        "name": "Cruz Central",
        "description": "Von Neumann r=1 (4 vecinos), corona r=2-4, 4 dendritas inh.",
        "center": "Von Neumann r=1 (4 vecinos)",
        "corona": "r=2-4, 4 bloques cardinales",
        "dendrites_inh": 4,
        "mask": MASK_CROSS_CENTER,
    },
    "one_dendrite": {
        "id": "one_dendrite",
        "name": "Una Dendrita",
        "description": "Moore r=1, corona r=2-4 en 1 sola dendrita inh.",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=2-4, todo en 1 dendrita",
        "dendrites_inh": 1,
        "mask": MASK_ONE_DENDRITE,
    },
    "fine_grain": {
        "id": "fine_grain",
        "name": "Grano Fino",
        "description": "Moore r=1, corona r=2-4, 16 sectores inh.",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=2-4, 16 sectores",
        "dendrites_inh": 16,
        "mask": MASK_FINE_GRAIN,
    },
    "double_ring": {
        "id": "double_ring",
        "name": "Doble Anillo",
        "description": "Moore r=1, anillo r=2-3 (-1) + anillo r=5-7 (-0.5).",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=2-3 (-1) + r=5-7 (-0.5)",
        "dendrites_inh": 16,
        "mask": MASK_DOUBLE_RING,
    },
    "soft_inhibit": {
        "id": "soft_inhibit",
        "name": "Inhibicion Suave",
        "description": "Moore r=1, corona r=2-4, peso inh. -0.5.",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=2-4, peso -0.5",
        "dendrites_inh": 8,
        "mask": MASK_SOFT_INHIBIT,
    },
    "strong_center": {
        "id": "strong_center",
        "name": "Centro Fuerte",
        "description": "Moore r=1 x2 dendritas exc., corona r=2-4.",
        "center": "Moore r=1 (2 dendritas exc.)",
        "corona": "r=2-4, peso -1",
        "dendrites_inh": 8,
        "mask": MASK_STRONG_CENTER,
    },
    "gradual_center": {
        "id": "gradual_center",
        "name": "Centro Gradual",
        "description": "Exc. gradual r=1(1.0) r=2(0.6) r=3(0.3), gap 2px, inh. sparse r=6-11.",
        "center": "Gradual r=1→1.0, r=2→0.6, r=3→0.3",
        "corona": "r=6-11, checkerboard sparse",
        "dendrites_inh": 8,
        "mask": MASK_GRADUAL_CENTER,
    },
    "gradual_big_inh": {
        "id": "gradual_big_inh",
        "name": "Centro Gradual Big Inh",
        "description": "Exc. gradual r=1-3, gap 4px, inh. sparse r=8-19.",
        "center": "Gradual r=1→1.0, r=2→0.6, r=3→0.3",
        "corona": "r=8-19, sparse step=3",
        "dendrites_inh": 8,
        "mask": MASK_GRADUAL_BIG_INH,
    },
    "gradual_xxl_inh": {
        "id": "gradual_xxl_inh",
        "name": "Centro Gradual XXL Inh",
        "description": "Exc. gradual r=1-3, gap 4px, inh. sparse r=8-30.",
        "center": "Gradual r=1→1.0, r=2→0.6, r=3→0.3",
        "corona": "r=8-30, sparse step=4",
        "dendrites_inh": 8,
        "mask": MASK_GRADUAL_XXL_INH,
    },
}


def get_mask(mask_id: str) -> MaskDef:
    """Get a mask definition by its ID. Raises KeyError if not found."""
    return MASK_PRESETS[mask_id]["mask"]


def _compute_preview_grid(mask: MaskDef) -> list[list[float | None]]:
    """Compute a 19×19 preview grid for a mask definition.

    Center is at (9, 9). The center cell is marked with 999.0 (inspected-cell
    convention). Each offset (dx, dy) maps to col=9+dx, row=9+dy. If two
    dendrites overlap on the same cell the one with the larger absolute weight
    wins.
    """
    size = 19
    center = 9
    grid: list[list[float | None]] = [[None] * size for _ in range(size)]
    grid[center][center] = 999.0

    for dendrite in mask:
        peso: float = dendrite["peso_dendrita"]
        for dx, dy in dendrite["offsets"]:
            col = center + dx
            row = center + dy
            if 0 <= row < size and 0 <= col < size:
                existing = grid[row][col]
                if existing is None or abs(peso) > abs(existing):
                    grid[row][col] = peso

    return grid


def _compute_mask_stats(mask: MaskDef) -> dict[str, Any]:
    """Compute static wiring stats for a mask definition.

    Returns per-neuron synapse counts and effective radii (Chebyshev distance).
    """
    exc_synapses = 0
    inh_synapses = 0
    max_exc_radius = 0
    max_inh_radius = 0

    for dendrite in mask:
        peso: float = dendrite["peso_dendrita"]
        offsets = dendrite["offsets"]
        n = len(offsets)
        max_r = max((max(abs(dx), abs(dy)) for dx, dy in offsets), default=0)
        if peso > 0:
            exc_synapses += n
            max_exc_radius = max(max_exc_radius, max_r)
        else:
            inh_synapses += n
            max_inh_radius = max(max_inh_radius, max_r)

    return {
        "excitatory_synapses": exc_synapses,
        "inhibitory_synapses": inh_synapses,
        "ratio_exc_inh": round(exc_synapses / max(inh_synapses, 1), 3),
        "excitation_radius": max_exc_radius,
        "inhibition_radius": max_inh_radius,
    }


def get_mask_info() -> list[dict[str, Any]]:
    """Get metadata for all mask presets (without the mask data itself)."""
    result = []
    for preset in MASK_PRESETS.values():
        entry = {k: v for k, v in preset.items() if k != "mask"}
        entry["preview_grid"] = _compute_preview_grid(preset["mask"])
        entry["mask_stats"] = _compute_mask_stats(preset["mask"])
        result.append(entry)
    return result

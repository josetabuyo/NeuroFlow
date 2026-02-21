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

# simple — original Mexican hat mask (Moore r=1 exc, 8 inhibitory blocks r=2-4)
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

MASK_BIG_CENTER_WIDE_INH: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(2)},
    *_make_inhibitory(_ring(4, 10), -1.0, 8),
]

MASK_SMALL_CENTER_GAP_WIDE_INH_X2: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    *_make_inhibitory(_ring(3, 13), -1.0, 8),
]

MASK_BIG_CENTER_WIDE_INH_X2: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(2)},
    *_make_inhibitory(_ring(4, 14), -1.0, 8),
]

MASK_XL_CENTER_GAP_WIDE_INH_X2: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(5, 15), -1.0, 8),
]

MASK_DEAMON_3_EN_50: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(5, 15), -1.0, 8),
]

MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_UP: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _shift(_moore(3), 0, 1)},
    *_make_inhibitory(_shift(_ring(5, 15), 0, 1), -1.0, 8),
]

MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_DOWN: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _shift(_moore(3), 0, -1)},
    *_make_inhibitory(_shift(_ring(5, 15), 0, -1), -1.0, 8),
]

MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_LEFT: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _shift(_moore(3), 1, 0)},
    *_make_inhibitory(_shift(_ring(5, 15), 1, 0), -1.0, 8),
]

MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_RIGHT: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _shift(_moore(3), -1, 0)},
    *_make_inhibitory(_shift(_ring(5, 15), -1, 0), -1.0, 8),
]

MASK_DEAMON_1_EN_50: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_sparse_ring(5, 35, step=2), -1.0, 8),
]

MASK_DEAMON_1_5_EN_50: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(7, 17), -1.0, 8),
]

MASK_DEAMON_1_5_EN_50_G6: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(10, 20), -1.0, 8),
]

MASK_DEAMON_E3_G12_I11: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(16, 26), -1.0, 8),
]

MASK_DEAMON_E3_G12_I5: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(16, 20), -1.0, 8),
]

MASK_DEAMON_E1_G12_I1: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(1)},
    *_make_inhibitory(_ring(14, 14), -1.0, 8),
]

MASK_DEAMON_E2_G12_I1: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(2)},
    *_make_inhibitory(_ring(15, 15), -1.0, 8),
]

MASK_DEAMON_E3_G12_I1: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(16, 16), -1.0, 8),
]

MASK_DEAMON_E3_G12_I3: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_ring(16, 18), -1.0, 8),
]

MASK_DEAMON_E3_G12_I3_DE3_DI3: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _random_sparse(_moore(3), 1 / 3, seed=42)},
    *_make_inhibitory(_random_sparse(_ring(16, 18), 1 / 3, seed=43), -1.0, 8),
]

MASK_DEAMON_E3_G12_I3_DE1_DI3: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_random_sparse(_ring(16, 18), 1 / 3, seed=43), -1.0, 8),
]

MASK_DEAMON_E3_G12_I3_DE3_DI1: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _random_sparse(_moore(3), 1 / 3, seed=42)},
    *_make_inhibitory(_ring(16, 18), -1.0, 8),
]

MASK_DEAMON_E2_G3_I3_DE1_5_DI1_5: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _random_sparse(_moore(2), 2 / 3, seed=44)},
    *_make_inhibitory(_random_sparse(_ring(6, 8), 2 / 3, seed=45), -1.0, 8),
]

MASK_DEAMON_E2_G3_I3_DE1_DI1_5: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(2)},
    *_make_inhibitory(_random_sparse(_ring(6, 8), 2 / 3, seed=45), -1.0, 8),
]

MASK_DEAMON_E2_G3_I3_DE1_DI1: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(2)},
    *_make_inhibitory(_ring(6, 8), -1.0, 8),
]

MASK_DEAMON_E2_G6_I3_DE1_DI1: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(2)},
    *_make_inhibitory(_ring(9, 11), -1.0, 8),
]

MASK_DEAMON_E3_G8_I3_DE1_DI1_1: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _moore(3)},
    *_make_inhibitory(_random_sparse(_ring(12, 14), 1 / 1.1, seed=43), -1.0, 8),
]

MASK_BIG_CENTER_SOFT_WIDE_INH: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _ring(1, 1)},
    {"peso_dendrita": 0.7, "offsets": _ring(2, 2)},
    *_make_inhibitory(_ring(4, 10), -1.0, 8),
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

MASK_GRADUAL_XXL_INH_SMALL: MaskDef = [
    {"peso_dendrita": 1.0, "offsets": _ring(1, 1)},
    *_make_inhibitory(_sparse_ring(5, 30, step=4), -1.0, 8),
]

MASK_MEXICAN_HAT: MaskDef = [
    # Excitatory peak — sharp falloff
    {"peso_dendrita": 1.0, "offsets": _ring(1, 1)},
    {"peso_dendrita": 0.5, "offsets": _ring(2, 2)},
    # Inhibitory profile — strong near center, decays with distance
    *_make_inhibitory(_ring(3, 5), -1.0, 8),
    *_make_inhibitory(_sparse_ring(6, 12, step=2), -0.6, 8),
    *_make_inhibitory(_sparse_ring(13, 20, step=3), -0.25, 8),
    *_make_inhibitory(_sparse_ring(21, 30, step=5), -0.08, 8),
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
        "name": "Deamon 3 en 50",
        "description": "Moore r=3 (48 vecinos), gap r=4, corona r=5-15 (misma superficie inh que x2).",
        "center": "Moore r=3 (48 vecinos)",
        "corona": "r=5-15, gap r=4 silencio",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_3_EN_50,
    },
    "deamon_e3_g1_i11_de1_di1_move_up": {
        "id": "deamon_e3_g1_i11_de1_di1_move_up",
        "name": "Deamon (E3 G1 I11 DE1 DI1) Move Up",
        "description": "Moore r=3 (48 vecinos) desplazado +1 dy, gap r=4, corona r=5-15 desplazada +1 dy.",
        "center": "Moore r=3 (48 vecinos, desplazado 1px abajo)",
        "corona": "r=5-15 desplazada 1px abajo, gap r=4 silencio",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_UP,
    },
    "deamon_e3_g1_i11_de1_di1_move_down": {
        "id": "deamon_e3_g1_i11_de1_di1_move_down",
        "name": "Deamon (E3 G1 I11 DE1 DI1) Move Down",
        "description": "Moore r=3 (48 vecinos) desplazado -1 dy, gap r=4, corona r=5-15 desplazada -1 dy.",
        "center": "Moore r=3 (48 vecinos, desplazado 1px arriba)",
        "corona": "r=5-15 desplazada 1px arriba, gap r=4 silencio",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_DOWN,
    },
    "deamon_e3_g1_i11_de1_di1_move_left": {
        "id": "deamon_e3_g1_i11_de1_di1_move_left",
        "name": "Deamon (E3 G1 I11 DE1 DI1) Move Left",
        "description": "Moore r=3 (48 vecinos) desplazado +1 dx, gap r=4, corona r=5-15 desplazada +1 dx.",
        "center": "Moore r=3 (48 vecinos, desplazado 1px a la derecha)",
        "corona": "r=5-15 desplazada 1px a la derecha, gap r=4 silencio",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_LEFT,
    },
    "deamon_e3_g1_i11_de1_di1_move_right": {
        "id": "deamon_e3_g1_i11_de1_di1_move_right",
        "name": "Deamon (E3 G1 I11 DE1 DI1) Move Right",
        "description": "Moore r=3 (48 vecinos) desplazado -1 dx, gap r=4, corona r=5-15 desplazada -1 dx.",
        "center": "Moore r=3 (48 vecinos, desplazado 1px a la izquierda)",
        "corona": "r=5-15 desplazada 1px a la izquierda, gap r=4 silencio",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G1_I11_DE1_DI1_MOVE_RIGHT,
    },
    "deamon_1_en_50": {
        "id": "deamon_1_en_50",
        "name": "Deamon 1 en 50",
        "description": "Moore r=3 (48 vecinos), gap r=4, corona sparse r=5-35 (inhibición masiva, ~1 deamon en 50x50).",
        "center": "Moore r=3 (48 vecinos)",
        "corona": "r=5-35 sparse step=2, gap r=4 silencio",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_1_EN_50,
    },
    "deamon_1_5_en_50": {
        "id": "deamon_1_5_en_50",
        "name": "Deamon 1,5 en 50 (E3 G3 I11)",
        "description": "Moore r=3 (48 vecinos), gap x3 r=4-6, corona r=7-17. ~1,5 deamons en 50x50.",
        "center": "Moore r=3 (48 vecinos)",
        "corona": "r=7-17, gap r=4-6 silencio (x3)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_1_5_EN_50,
    },
    "deamon_1_5_en_50_g6": {
        "id": "deamon_1_5_en_50_g6",
        "name": "Deamon 1,5 en 50 (E3 G6 I11)",
        "description": "Moore r=3 (48 vecinos), gap x6 r=4-9, corona r=10-20. ~1,5 deamons en 50x50.",
        "center": "Moore r=3 (48 vecinos)",
        "corona": "r=10-20, gap r=4-9 silencio (x6)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_1_5_EN_50_G6,
    },
    "deamon_e3_g12_i11": {
        "id": "deamon_e3_g12_i11",
        "name": "Deamon (E3 G12 I11)",
        "description": "Moore r=3 (48 vecinos), gap x12 r=4-15, corona r=16-26.",
        "center": "Moore r=3 (48 vecinos)",
        "corona": "r=16-26, gap r=4-15 silencio (x12)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G12_I11,
    },
    "deamon_e3_g12_i5": {
        "id": "deamon_e3_g12_i5",
        "name": "Deamon (E3 G12 I5)",
        "description": "Moore r=3 (48 vecinos), gap x12 r=4-15, corona r=16-20 (mitad de I11).",
        "center": "Moore r=3 (48 vecinos)",
        "corona": "r=16-20, gap r=4-15 silencio (x12)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G12_I5,
    },
    "deamon_e1_g12_i1": {
        "id": "deamon_e1_g12_i1",
        "name": "Deamon (E1 G12 I1)",
        "description": "Moore r=1 (8 vecinos), gap x12 r=2-13, corona r=14 (1 anillo).",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=14, gap r=2-13 silencio (x12)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E1_G12_I1,
    },
    "deamon_e2_g12_i1": {
        "id": "deamon_e2_g12_i1",
        "name": "Deamon (E2 G12 I1)",
        "description": "Moore r=2 (24 vecinos), gap x12 r=3-14, corona r=15 (1 anillo).",
        "center": "Moore r=2 (24 vecinos)",
        "corona": "r=15, gap r=3-14 silencio (x12)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E2_G12_I1,
    },
    "deamon_e3_g12_i1": {
        "id": "deamon_e3_g12_i1",
        "name": "Deamon (E3 G12 I1)",
        "description": "Moore r=3 (48 vecinos), gap x12 r=4-15, corona r=16 (1 anillo).",
        "center": "Moore r=3 (48 vecinos)",
        "corona": "r=16, gap r=4-15 silencio (x12)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G12_I1,
    },
    "deamon_e3_g12_i3": {
        "id": "deamon_e3_g12_i3",
        "name": "Deamon (E3 G12 I3)",
        "description": "Moore r=3 (48 vecinos), gap x12 r=4-15, corona r=16-18 (3 anillos).",
        "center": "Moore r=3 (48 vecinos)",
        "corona": "r=16-18, gap r=4-15 silencio (x12)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G12_I3,
    },
    "deamon_e3_g12_i3_de3_di3": {
        "id": "deamon_e3_g12_i3_de3_di3",
        "name": "Deamon (E3 G12 I3 DE3 DI3)",
        "description": "E3 G12 I3 con densidad 1/3 en exc. e inh.: ~33% de sinapsis al azar en ambas zonas.",
        "center": "Moore r=3 sparse ~33% (~16 vecinos)",
        "corona": "r=16-18 sparse ~33%, gap r=4-15 silencio (x12)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G12_I3_DE3_DI3,
    },
    "deamon_e3_g12_i3_de1_di3": {
        "id": "deamon_e3_g12_i3_de1_di3",
        "name": "Deamon (E3 G12 I3 DE1 DI3)",
        "description": "E3 G12 I3 con exc. completa (48 vecinos) e inh. sparse ~33%.",
        "center": "Moore r=3 (48 vecinos, densidad completa)",
        "corona": "r=16-18 sparse ~33%, gap r=4-15 silencio (x12)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G12_I3_DE1_DI3,
    },
    "deamon_e3_g12_i3_de3_di1": {
        "id": "deamon_e3_g12_i3_de3_di1",
        "name": "Deamon (E3 G12 I3 DE3 DI1)",
        "description": "E3 G12 I3 con exc. sparse ~33% e inh. completa (3 anillos).",
        "center": "Moore r=3 sparse ~33% (~16 vecinos)",
        "corona": "r=16-18 completa, gap r=4-15 silencio (x12)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G12_I3_DE3_DI1,
    },
    "deamon_e2_g3_i3_de1_5_di1_5": {
        "id": "deamon_e2_g3_i3_de1_5_di1_5",
        "name": "Deamon (E2 G3 I3 DE1.5 DI1.5)",
        "description": "Moore r=2 sparse ~67%, gap r=3-5, corona r=6-8 sparse ~67%.",
        "center": "Moore r=2 sparse ~67% (~16 vecinos)",
        "corona": "r=6-8 sparse ~67%, gap r=3-5 silencio (x3)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E2_G3_I3_DE1_5_DI1_5,
    },
    "deamon_e2_g3_i3_de1_di1_5": {
        "id": "deamon_e2_g3_i3_de1_di1_5",
        "name": "Deamon (E2 G3 I3 DE1 DI1.5)",
        "description": "Moore r=2 completa (24 vecinos), gap r=3-5, corona r=6-8 sparse ~67%.",
        "center": "Moore r=2 (24 vecinos, densidad completa)",
        "corona": "r=6-8 sparse ~67%, gap r=3-5 silencio (x3)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E2_G3_I3_DE1_DI1_5,
    },
    "deamon_e2_g3_i3_de1_di1": {
        "id": "deamon_e2_g3_i3_de1_di1",
        "name": "Deamon (E2 G3 I3 DE1 DI1)",
        "description": "Moore r=2 completa (24 vecinos), gap r=3-5, corona r=6-8 completa.",
        "center": "Moore r=2 (24 vecinos, densidad completa)",
        "corona": "r=6-8 completa, gap r=3-5 silencio (x3)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E2_G3_I3_DE1_DI1,
    },
    "deamon_e2_g6_i3_de1_di1": {
        "id": "deamon_e2_g6_i3_de1_di1",
        "name": "Deamon (E2 G6 I3 DE1 DI1)",
        "description": "Moore r=2 completa (24 vecinos), gap r=3-8, corona r=9-11 completa.",
        "center": "Moore r=2 (24 vecinos, densidad completa)",
        "corona": "r=9-11 completa, gap r=3-8 silencio (x6)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E2_G6_I3_DE1_DI1,
    },
    "deamon_e3_g8_i3_de1_di1_1": {
        "id": "deamon_e3_g8_i3_de1_di1_1",
        "name": "Deamon (E3 G8 I3 DE1 DI1.1)",
        "description": "Moore r=3 completa (48 vecinos), gap r=4-11, corona r=12-14 sparse ~91%.",
        "center": "Moore r=3 (48 vecinos, densidad completa)",
        "corona": "r=12-14 sparse ~91%, gap r=4-11 silencio (x8)",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_DEAMON_E3_G8_I3_DE1_DI1_1,
    },
    "all_exc": {
        "id": "all_exc",
        "name": "Todo Exc",
        "description": "1 dendrita exc. r=1 (8 vecinos).",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "sin inhibición",
        "dendrites_inh": 0,
        "random_weights": True,
        "mask": MASK_ALL_EXC,
    },
    "all_inh": {
        "id": "all_inh",
        "name": "Todo Inh",
        "description": "1 dendrita inh. r=1 (8 vecinos).",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "sin excitación",
        "dendrites_inh": 1,
        "random_weights": True,
        "mask": MASK_ALL_INH,
    },
    "simple": {
        "id": "simple",
        "name": "Kohonen Simple",
        "description": "Moore r=1, corona r=2-4, 8 dendritas inh.",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=2-4, 8 bloques 3x3",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_SIMPLE,
    },
    "wide_hat": {
        "id": "wide_hat",
        "name": "Sombrero Ancho",
        "description": "Moore r=1, corona r=2-7, 8 dendritas inh.",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=2-7, corona grande",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_WIDE_HAT,
    },
    "narrow_hat": {
        "id": "narrow_hat",
        "name": "Sombrero Estrecho",
        "description": "Moore r=1, corona r=2-3, 8 dendritas inh.",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=2-3, corona cercana",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_NARROW_HAT,
    },
    "big_center": {
        "id": "big_center",
        "name": "Centro Grande",
        "description": "Moore r=2 (24 vecinos), corona r=4-7, 8 dendritas inh.",
        "center": "Moore r=2 (24 vecinos)",
        "corona": "r=4-7, corona lejana",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_BIG_CENTER,
    },
    "big_center_wide_inh": {
        "id": "big_center_wide_inh",
        "name": "Centro Grande Inh Ancha",
        "description": "Moore r=2 (24 vecinos), corona r=4-10 (2x superficie inh).",
        "center": "Moore r=2 (24 vecinos)",
        "corona": "r=4-10, corona extendida",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_BIG_CENTER_WIDE_INH,
    },
    "small_center_gap_wide_inh_x2": {
        "id": "small_center_gap_wide_inh_x2",
        "name": "Centro Chico + Gap + Inh Ancha x2",
        "description": "Moore r=1 (8 vecinos), gap r=2, corona r=3-13 (misma superficie inh que x2).",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=3-13, gap r=2 silencio",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_SMALL_CENTER_GAP_WIDE_INH_X2,
    },
    "big_center_wide_inh_x2": {
        "id": "big_center_wide_inh_x2",
        "name": "Centro Grande Inh Ancha x2",
        "description": "Moore r=2 (24 vecinos), corona r=4-14 (4x superficie inh original).",
        "center": "Moore r=2 (24 vecinos)",
        "corona": "r=4-14, corona muy extendida",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_BIG_CENTER_WIDE_INH_X2,
    },
    "xl_center_gap_wide_inh_x2": {
        "id": "xl_center_gap_wide_inh_x2",
        "name": "Centro XL + Gap + Inh Ancha x2",
        "description": "Moore r=3 (48 vecinos), gap r=4, corona r=5-15 (misma superficie inh que x2).",
        "center": "Moore r=3 (48 vecinos)",
        "corona": "r=5-15, gap r=4 silencio",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_XL_CENTER_GAP_WIDE_INH_X2,
    },
    "big_center_soft_wide_inh": {
        "id": "big_center_soft_wide_inh",
        "name": "Centro Grande Soft Inh Ancha",
        "description": "Exc. r=1(1.0) r=2(0.7) borde suave, corona r=4-10.",
        "center": "r=1→1.0, r=2→0.7 (borde suave)",
        "corona": "r=4-10, corona extendida",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_BIG_CENTER_SOFT_WIDE_INH,
    },
    "cross_center": {
        "id": "cross_center",
        "name": "Cruz Central",
        "description": "Von Neumann r=1 (4 vecinos), corona r=2-4, 4 dendritas inh.",
        "center": "Von Neumann r=1 (4 vecinos)",
        "corona": "r=2-4, 4 bloques cardinales",
        "dendrites_inh": 4,
        "random_weights": True,
        "mask": MASK_CROSS_CENTER,
    },
    "one_dendrite": {
        "id": "one_dendrite",
        "name": "Una Dendrita",
        "description": "Moore r=1, corona r=2-4 en 1 sola dendrita inh.",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=2-4, todo en 1 dendrita",
        "dendrites_inh": 1,
        "random_weights": True,
        "mask": MASK_ONE_DENDRITE,
    },
    "fine_grain": {
        "id": "fine_grain",
        "name": "Grano Fino",
        "description": "Moore r=1, corona r=2-4, 16 sectores inh.",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=2-4, 16 sectores",
        "dendrites_inh": 16,
        "random_weights": True,
        "mask": MASK_FINE_GRAIN,
    },
    "double_ring": {
        "id": "double_ring",
        "name": "Doble Anillo",
        "description": "Moore r=1, anillo r=2-3 (-1) + anillo r=5-7 (-0.5).",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=2-3 (-1) + r=5-7 (-0.5)",
        "dendrites_inh": 16,
        "random_weights": True,
        "mask": MASK_DOUBLE_RING,
    },
    "soft_inhibit": {
        "id": "soft_inhibit",
        "name": "Inhibicion Suave",
        "description": "Moore r=1, corona r=2-4, peso inh. -0.5.",
        "center": "Moore r=1 (8 vecinos)",
        "corona": "r=2-4, peso -0.5",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_SOFT_INHIBIT,
    },
    "strong_center": {
        "id": "strong_center",
        "name": "Centro Fuerte",
        "description": "Moore r=1 x2 dendritas exc., corona r=2-4.",
        "center": "Moore r=1 (2 dendritas exc.)",
        "corona": "r=2-4, peso -1",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_STRONG_CENTER,
    },
    "gradual_center": {
        "id": "gradual_center",
        "name": "Centro Gradual",
        "description": "Exc. gradual r=1(1.0) r=2(0.6) r=3(0.3), gap 2px, inh. sparse r=6-11.",
        "center": "Gradual r=1→1.0, r=2→0.6, r=3→0.3",
        "corona": "r=6-11, checkerboard sparse",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_GRADUAL_CENTER,
    },
    "gradual_big_inh": {
        "id": "gradual_big_inh",
        "name": "Centro Gradual Big Inh",
        "description": "Exc. gradual r=1-3, gap 4px, inh. sparse r=8-19.",
        "center": "Gradual r=1→1.0, r=2→0.6, r=3→0.3",
        "corona": "r=8-19, sparse step=3",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_GRADUAL_BIG_INH,
    },
    "gradual_xxl_inh": {
        "id": "gradual_xxl_inh",
        "name": "Centro Gradual XXL Inh",
        "description": "Exc. gradual r=1-3, gap 4px, inh. sparse r=8-30.",
        "center": "Gradual r=1→1.0, r=2→0.6, r=3→0.3",
        "corona": "r=8-30, sparse step=4",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_GRADUAL_XXL_INH,
    },
    "gradual_xxl_inh_small": {
        "id": "gradual_xxl_inh_small",
        "name": "Centro Chico XXL Inh",
        "description": "Exc. solo r=1, gap 3px, inh. sparse r=5-30.",
        "center": "r=1→1.0 (solo vecinos inmediatos)",
        "corona": "r=5-30, sparse step=4",
        "dendrites_inh": 8,
        "random_weights": True,
        "mask": MASK_GRADUAL_XXL_INH_SMALL,
    },
    "mexican_hat": {
        "id": "mexican_hat",
        "name": "Sombrero Mexicano",
        "description": "DoG teórico: exc. r=1-2, inh. gradual r=3-30 (peso decae con distancia).",
        "center": "Gradual r=1→1.0, r=2→0.5",
        "corona": "r=3-5(-1) → r=6-12(-0.6) → r=13-20(-0.25) → r=21-30(-0.08)",
        "dendrites_inh": 32,
        "random_weights": True,
        "mask": MASK_MEXICAN_HAT,
    },
    "rule_110": {
        "id": "rule_110",
        "name": "Wolfram Rule 110",
        "description": "Autómata celular elemental Rule 110 (Turing-completo).",
        "center": "3 vecinos fila inferior",
        "corona": "sin inhibición",
        "dendrites_inh": 0,
        "random_weights": False,
        "mask_type": "wolfram",
        "mask": MASK_RULE_110,
    },
    "rule_30": {
        "id": "rule_30",
        "name": "Wolfram Rule 30",
        "description": "Autómata celular elemental Rule 30 (caótico, pseudo-random).",
        "center": "3 vecinos fila inferior",
        "corona": "sin inhibición",
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


def _compute_preview_grid(mask: MaskDef) -> list[list[float | None]]:
    """Compute a preview grid sized to fit 100% of the mask.

    The grid is (2*max_r+1) square, with the center cell marked as 999.0.
    Each offset (dx, dy) maps to col=center+dx, row=center+dy. If two
    dendrites overlap on the same cell the one with the larger absolute weight
    wins.
    """
    max_r = 0
    for dendrite in mask:
        for dx, dy in dendrite["offsets"]:
            max_r = max(max_r, abs(dx), abs(dy))

    max_r = max(max_r, 1)
    size = 2 * max_r + 1
    center = max_r
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

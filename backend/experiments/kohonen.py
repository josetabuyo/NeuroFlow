"""Experimento Kohonen — competencia lateral 2D (mapa autoorganizado).

Implementa el conexionado `kohonen_simple` del modelo original:
- 1 dendrita excitatoria (vecinos inmediatos, Moore neighborhood)
- 8 dendritas inhibitorias (bloques 3×3 a distancia ~3, anillo octagonal)

Cada neurona excita a sus vecinas cercanas e inhibe a las lejanas.
El perfil "Mexican hat" produce clusters de actividad que compiten entre sí.
"""

from __future__ import annotations

import random
from typing import Any

from core.constructor import Constructor
from .base import Experimento


# ---------------------------------------------------------------------------
# Máscara kohonen_simple — offsets exactos del proyecto JS original
# ---------------------------------------------------------------------------

KOHONEN_SIMPLE_MASK: list[dict[str, object]] = [
    # D0 — Excitatoria: 8 vecinos inmediatos (Moore neighborhood)
    {
        "peso_dendrita": 1.0,
        "offsets": [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ],
    },
    # D1 — Inhibitoria NE: bloque 3×3 en (+2..+4, -4..-2)
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (2, -4), (2, -3), (2, -2),
            (3, -4), (3, -3), (3, -2),
            (4, -4), (4, -3), (4, -2),
        ],
    },
    # D2 — Inhibitoria E: bloque 3×3 en (+2..+4, -1..+1)
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (2, -1), (2, 0), (2, 1),
            (3, -1), (3, 0), (3, 1),
            (4, -1), (4, 0), (4, 1),
        ],
    },
    # D3 — Inhibitoria SE: bloque 3×3 en (+2..+4, +2..+4)
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (2, 2), (2, 3), (2, 4),
            (3, 2), (3, 3), (3, 4),
            (4, 2), (4, 3), (4, 4),
        ],
    },
    # D4 — Inhibitoria S: bloque 3×3 en (-1..+1, +2..+4)
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (-1, 2), (-1, 3), (-1, 4),
            (0, 2),  (0, 3),  (0, 4),
            (1, 2),  (1, 3),  (1, 4),
        ],
    },
    # D5 — Inhibitoria SW: bloque 3×3 en (-4..-2, +2..+4)
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (-4, 2), (-4, 3), (-4, 4),
            (-3, 2), (-3, 3), (-3, 4),
            (-2, 2), (-2, 3), (-2, 4),
        ],
    },
    # D6 — Inhibitoria W: bloque 3×3 en (-4..-2, -1..+1)
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (-4, -1), (-4, 0), (-4, 1),
            (-3, -1), (-3, 0), (-3, 1),
            (-2, -1), (-2, 0), (-2, 1),
        ],
    },
    # D7 — Inhibitoria NW: bloque 3×3 en (-4..-2, -4..-2)
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (-4, -4), (-4, -3), (-4, -2),
            (-3, -4), (-3, -3), (-3, -2),
            (-2, -4), (-2, -3), (-2, -2),
        ],
    },
    # D8 — Inhibitoria N: bloque 3×3 en (-1..+1, -4..-2)
    {
        "peso_dendrita": -1.0,
        "offsets": [
            (-1, -4), (-1, -3), (-1, -2),
            (0, -4),  (0, -3),  (0, -2),
            (1, -4),  (1, -3),  (1, -2),
        ],
    },
]


class KohonenExperiment(Experimento):
    """Mapa autoorganizado de Kohonen con competencia lateral 2D."""

    def __init__(self) -> None:
        super().__init__()
        self._config: dict[str, Any] = {}

    def setup(self, config: dict[str, Any]) -> None:
        """Configura el experimento: grilla 2D con máscara kohonen_simple."""
        self._config = config
        self.width = config.get("width", 30)
        self.height = config.get("height", 30)
        self.generation = 0

        constructor = Constructor()

        # Todas las neuronas son internas (sin entrada ni salida), umbral = 0.0
        self.red, self.regiones = constructor.crear_grilla(
            width=self.width,
            height=self.height,
            filas_entrada=[],
            filas_salida=[],
            umbral=0.0,
        )

        # Aplicar máscara de conexión kohonen_simple
        constructor.aplicar_mascara_2d(
            self.red, self.width, self.height, KOHONEN_SIMPLE_MASK
        )

        # Inicializar todas las neuronas con valores aleatorios
        for neurona in self.red.neuronas:
            neurona.activar_external(random.random())

    def click(self, x: int, y: int) -> None:
        """Toggle: si valor < 0.5 → activar (1.0), si >= 0.5 → desactivar (0.0)."""
        key = Constructor.key_by_coord(x, y)
        try:
            neurona = self.red.get_neurona(key)
            if neurona.valor < 0.5:
                neurona.activar_external(1.0)
            else:
                neurona.activar_external(0.0)
        except KeyError:
            pass

    def step(self) -> dict[str, Any]:
        """Un step procesa toda la red de golpe. Kohonen nunca termina."""
        self.red.procesar()
        self.generation += 1

        frame = self.get_frame()
        stats = self.get_stats()

        return {
            "type": "frame",
            "generation": self.generation,
            "grid": frame,
            "stats": stats,
        }

    def reset(self) -> None:
        """Reinicia el experimento con nuevos valores aleatorios."""
        self.setup(self._config)

    def is_complete(self) -> bool:
        """Kohonen nunca termina."""
        return False

"""Tests for the Connection Inspector — inspect() in Experiment.

Validates:
- weight_grid with correct dimensions
- Excitatory neighbors with positive weight
- Inhibitory blocks with negative weight
- Cells without connection -> None
- Inspected cell marked with 999
- Neurona on border: same connections as center (toroidal topology)
- total_dendritas and total_sinapsis correct
- Effective weights clamped to [-1, 1]
- Source neuron in multiple dendrites: weights summed
"""

import random

import pytest
from core.constructor import Constructor
from core.sinapsis import Sinapsis
from core.dendrita import Dendrita
from core.neurona import Neurona, NeuronaEntrada
from core.brain import Brain
from core.masks import MASK_SIMPLE
from experiments.experiment import Experiment


def _nested_config(
    width: int = 10, height: int = 10, mask: str = "simple",
) -> dict:
    return {
        "grid": {"width": width, "height": height},
        "wiring": {"mask": mask, "process_mode": "min_vs_max"},
    }


class TestInspect:
    """Experiment.inspect() returns effective weight map."""

    def test_inspect_retorna_weight_grid_dimensiones_correctas(self) -> None:
        random.seed(42)
        exp = Experiment()
        exp.setup(_nested_config())

        result = exp.inspect(5, 5)

        assert result["type"] == "connections"
        assert result["x"] == 5
        assert result["y"] == 5
        grid = result["weight_grid"]
        assert len(grid) == 10
        assert all(len(row) == 10 for row in grid)

    def test_vecinos_excitatorios_peso_positivo(self) -> None:
        random.seed(42)
        exp = Experiment()
        exp.setup(_nested_config())

        result = exp.inspect(5, 5)
        grid = result["weight_grid"]

        vecinos = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in vecinos:
            nx, ny = 5 + dx, 5 + dy
            peso = grid[ny][nx]
            assert peso is not None, f"Neighbor ({nx},{ny}) should be connected"
            assert peso > 0, f"Neighbor ({nx},{ny}) should be excitatory, but has weight {peso}"

    def test_bloques_inhibitorios_peso_negativo(self) -> None:
        random.seed(42)
        exp = Experiment()
        exp.setup(_nested_config())

        result = exp.inspect(5, 5)
        grid = result["weight_grid"]

        offsets_d2 = [(2, -1), (2, 0), (2, 1), (3, -1), (3, 0), (3, 1), (4, -1), (4, 0), (4, 1)]
        for dx, dy in offsets_d2:
            nx, ny = 5 + dx, 5 + dy
            if 0 <= nx < 10 and 0 <= ny < 10:
                peso = grid[ny][nx]
                assert peso is not None, f"Cell ({nx},{ny}) should be connected"
                assert peso < 0, f"Cell ({nx},{ny}) should be inhibitory, but has weight {peso}"

    def test_celda_sin_conexion_es_none(self) -> None:
        random.seed(42)
        exp = Experiment()
        exp.setup(_nested_config(width=30, height=30))

        result = exp.inspect(15, 15)
        grid = result["weight_grid"]

        assert grid[0][0] is None
        assert grid[29][29] is None

    def test_celda_inspeccionada_marcada_999(self) -> None:
        random.seed(42)
        exp = Experiment()
        exp.setup(_nested_config())

        result = exp.inspect(5, 5)
        grid = result["weight_grid"]

        assert grid[5][5] == 999

    def test_neurona_borde_mismas_conexiones_que_centro(self) -> None:
        random.seed(42)
        exp = Experiment()
        exp.setup(_nested_config(width=30, height=30))

        result_centro = exp.inspect(15, 15)
        result_borde = exp.inspect(0, 0)

        grid_centro = result_centro["weight_grid"]
        grid_borde = result_borde["weight_grid"]

        conexiones_centro = sum(
            1 for row in grid_centro for cell in row if cell is not None and cell != 999
        )
        conexiones_borde = sum(
            1 for row in grid_borde for cell in row if cell is not None and cell != 999
        )

        assert conexiones_borde == conexiones_centro

    def test_total_dendritas_y_sinapsis_correctos(self) -> None:
        random.seed(42)
        exp = Experiment()
        exp.setup(_nested_config())

        result = exp.inspect(5, 5)

        assert result["total_dendritas"] == 13

        neurona = exp.brain.get_neurona("x5y5")
        total_sinapsis = sum(len(d.sinapsis) for d in neurona.dendritas)
        assert result["total_sinapsis"] == total_sinapsis

    def test_pesos_efectivos_clampeados(self) -> None:
        random.seed(42)
        exp = Experiment()
        exp.setup(_nested_config())

        result = exp.inspect(5, 5)
        grid = result["weight_grid"]

        for row in grid:
            for cell in row:
                if cell is not None and cell != 999:
                    assert -1.0 <= cell <= 1.0, f"Weight {cell} out of range [-1, 1]"

    def test_neurona_fuente_multiples_dendritas_suma(self) -> None:
        constructor = Constructor()
        brain, _ = constructor.crear_grilla(
            width=3, height=3, filas_entrada=[], filas_salida=[], umbral=0.0
        )

        neurona_destino = brain.get_neurona("x1y1")
        neurona_fuente = brain.get_neurona("x0y0")

        s1 = Sinapsis(neurona_entrante=neurona_fuente, peso=0.8)
        d1 = Dendrita(sinapsis=[s1], peso=1.0)

        s2 = Sinapsis(neurona_entrante=neurona_fuente, peso=0.5)
        d2 = Dendrita(sinapsis=[s2], peso=-1.0)

        neurona_destino.dendritas.extend([d1, d2])

        exp = Experiment()
        exp.brain = brain
        exp.width = 3
        exp.height = 3
        exp.regiones = {}
        exp.generation = 0

        result = exp.inspect(1, 1)
        grid = result["weight_grid"]

        assert grid[0][0] == pytest.approx(0.3, abs=1e-9)


# ---------------------------------------------------------------------------
# Nerve inspection: tissue daemon connections + nociceptor outgoing footprint
# ---------------------------------------------------------------------------

def _nerve_inspect_config() -> dict:
    """10×10 tissue with daemon + nerve at (5,5,r=3) from nociceptor to output."""
    return {
        "regions": [
            {"id": "tissue", "grid": {"width": 10, "height": 10}},
            {"id": "output", "grid": {"width": 1, "height": 1}},
            {"id": "nociceptor", "grid": {"width": 1, "height": 1}},
        ],
        "connections": [
            {
                "on": "tissue",
                "deamon": {
                    "shape": "square",
                    "fixed": True,
                    "groups": [
                        {"id": "excitatory", "weight": 0.9, "noise": 0, "weights": [1]},
                        {"id": "inhibitory", "weight": -1.0, "gap": 1, "noise": 0, "sectors": 4, "weights": [1]},
                    ],
                },
            },
            {
                "on": "tissue",
                "nerve": {
                    "insertion": {"x": 5, "y": 5},
                    "radius": 3,
                    "from": {"region": "nociceptor", "density": 1.0, "weight": -0.5},
                    "to": {"region": "output", "density": 1.0, "weight": 0.5},
                },
            },
        ],
    }


class TestInspectNerve:
    """inspect() correctness when a nerve connects regions via the wiring region."""

    def test_tissue_weight_grid_shows_daemon_connections(self) -> None:
        """Tissue inspection always returns daemon weight_grid even when nerve circles exist."""
        random.seed(42)
        exp = Experiment()
        exp.setup(_nerve_inspect_config())
        result = exp.inspect(5, 5)
        grid = result["weight_grid"]
        non_null = [v for row in grid for v in row if v is not None and v != 999]
        assert len(non_null) > 0, "Daemon connections must be visible in tissue weight_grid"
        # Excitatory neighbors are positive
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            v = grid[5 + dy][5 + dx]
            assert v is not None and v > 0

    def test_tissue_inspect_no_nociceptor_source_grid(self) -> None:
        """Tissue inspection must NOT return nociceptor_weight_grid (1×1 box is redundant/confusing)."""
        random.seed(42)
        exp = Experiment()
        exp.setup(_nerve_inspect_config())
        # Inspect a neuron inside the nerve circle — nociceptor has connections here
        result = exp.inspect(5, 5)
        assert "nociceptor_weight_grid" not in result, (
            "Nociceptor source weight grid must be excluded from tissue inspection"
        )

    def test_tissue_inspect_returns_nerve_circles(self) -> None:
        """Tissue inspection includes nerve circles so the UI can draw them."""
        random.seed(42)
        exp = Experiment()
        exp.setup(_nerve_inspect_config())
        result = exp.inspect(5, 5)
        assert "nerve_circles" in result
        circles = result["nerve_circles"]
        assert any(c["cx"] == 5 and c["cy"] == 5 and c["radius"] == 3 for c in circles)

    def test_nociceptor_shows_outgoing_footprint_on_tissue(self) -> None:
        """Nociceptor inspection returns weight_grid = outgoing footprint on the wiring region."""
        random.seed(42)
        exp = Experiment()
        exp.setup(_nerve_inspect_config())
        result = exp.inspect(0, 0, region_id="nociceptor")
        grid = result["weight_grid"]
        # Dimensions match tissue (10×10)
        assert len(grid) == 10
        assert all(len(row) == 10 for row in grid)
        # Some non-null values (neurons inside nerve circle at 5,5 r=3 receive from nociceptor)
        non_null = [v for row in grid for v in row if v is not None]
        assert len(non_null) > 0, "Nociceptor outgoing footprint must have non-null weights"
        # All negative (inhibitory weight=-0.5)
        assert all(v <= 0 for v in non_null), "Nociceptor→tissue connections are inhibitory"

    def test_nociceptor_inspect_returns_nerve_circles(self) -> None:
        """Nociceptor inspection includes nerve circles from the nerve it participates in."""
        random.seed(42)
        exp = Experiment()
        exp.setup(_nerve_inspect_config())
        result = exp.inspect(0, 0, region_id="nociceptor")
        assert "nerve_circles" in result
        circles = result["nerve_circles"]
        assert len(circles) >= 1
        assert circles[0]["cx"] == 5
        assert circles[0]["cy"] == 5
        assert circles[0]["radius"] == 3

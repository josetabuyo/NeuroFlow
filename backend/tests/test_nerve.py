"""Tests for the nerve connection type — spatial bundles of cross-region connections."""

import random

import pytest
from core.nerve import (
    NerveCircle,
    place_nerve_circles,
    circle_cells_with_weights,
)
from experiments.experiment import Experiment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


def _experiment_config(nerve_cfg: dict) -> dict:
    return {
        "regions": [
            {"id": "tissue", "grid": {"width": 30, "height": 30}},
            {"id": "noci", "grid": {"width": 2, "height": 2}},
            {"id": "out", "grid": {"width": 10, "height": 5}},
        ],
        "connections": [
            {
                "on": "tissue",
                "deamon": {
                    "shape": "square",
                    "excitatory": {"weight": 0.9, "offset": 1, "noise": 0, "weights": [1]},
                    "inhibitory": {"weight": -1.0, "offset": 3, "noise": 0, "sectors": 4, "weights": [1]},
                },
            },
            {"on": "tissue", "nerve": nerve_cfg},
        ],
    }


# ---------------------------------------------------------------------------
# Placement: exact position
# ---------------------------------------------------------------------------

class TestPlacePosition:
    def test_exact_coordinates(self) -> None:
        circles = place_nerve_circles({"insertion": {"x": 10, "y": 15}, "radius": 6}, 50, 50)
        assert len(circles) == 1
        assert circles[0].cx == 10
        assert circles[0].cy == 15
        assert circles[0].radius == 6

    def test_exact_position_in_bounds(self) -> None:
        circles = place_nerve_circles({"insertion": {"x": 8, "y": 8}, "radius": 4}, 30, 30)
        c = circles[0]
        assert 0 <= c.cx < 30
        assert 0 <= c.cy < 30

    def test_radius_preserved(self) -> None:
        circles = place_nerve_circles({"insertion": {"x": 20, "y": 20}, "radius": 9}, 50, 50)
        assert circles[0].radius == 9


# ---------------------------------------------------------------------------
# Placement: random
# ---------------------------------------------------------------------------

class TestPlaceRandom:
    def test_returns_one_circle(self) -> None:
        circles = place_nerve_circles({"insertion": "random", "radius": 4}, 50, 50, _rng())
        assert len(circles) == 1

    def test_within_bounds(self) -> None:
        r = 5
        circles = place_nerve_circles({"insertion": "random", "radius": r}, 40, 40, _rng())
        c = circles[0]
        assert r <= c.cx < 40 - r
        assert r <= c.cy < 40 - r

    def test_default_insertion_is_random(self) -> None:
        circles = place_nerve_circles({"radius": 4}, 50, 50, _rng())
        assert len(circles) == 1


# ---------------------------------------------------------------------------
# circle_cells_with_weights
# ---------------------------------------------------------------------------

class TestCircleCells:
    def test_center_weight_is_one(self) -> None:
        c = NerveCircle(cx=10, cy=10, radius=5)
        cells = circle_cells_with_weights(c, 30, 30)
        center = [(w, x, y) for x, y, w in cells if x == 10 and y == 10]
        assert len(center) == 1
        assert center[0][0] == pytest.approx(1.0)

    def test_weight_decreases_with_distance(self) -> None:
        c = NerveCircle(cx=10, cy=10, radius=6)
        cells = circle_cells_with_weights(c, 30, 30)
        cell_map = {(x, y): w for x, y, w in cells}
        w_center = cell_map[(10, 10)]
        w_near = cell_map.get((10, 11)) or cell_map.get((11, 10))
        w_edge = min(cell_map.values())
        assert w_center > (w_near or 0)
        assert w_center > w_edge

    def test_all_within_bounds(self) -> None:
        w, h = 20, 20
        c = NerveCircle(cx=10, cy=10, radius=5)
        cells = circle_cells_with_weights(c, w, h)
        for x, y, _ in cells:
            assert 0 <= x < w
            assert 0 <= y < h

    def test_unit_radius_cells(self) -> None:
        # radius=1 circle centered at (5,5): center + 4 cardinal (dist=1) = 5 cells
        # also diagonal cells at dist=sqrt(2)≈1.41 are OUTSIDE radius=1
        c = NerveCircle(cx=5, cy=5, radius=1)
        cells = circle_cells_with_weights(c, 20, 20)
        coords = {(x, y) for x, y, _ in cells}
        assert (5, 5) in coords   # center
        assert (5, 4) in coords   # up
        assert (5, 6) in coords   # down
        assert (4, 5) in coords   # left
        assert (6, 5) in coords   # right
        assert (4, 4) not in coords  # diagonal, dist=sqrt(2) > 1

    def test_all_weights_in_range(self) -> None:
        c = NerveCircle(cx=8, cy=8, radius=4)
        cells = circle_cells_with_weights(c, 20, 20)
        for _, _, w in cells:
            assert 0.0 <= w <= 1.0


# ---------------------------------------------------------------------------
# Integration: Experiment with nerve connections
# ---------------------------------------------------------------------------

_NERVE_CFG_INTEGRATION = {
    "insertion": {"x": 10, "y": 10},
    "radius": 3,
    "from": {"region": "noci", "density": 1.0, "weight": -0.01},
    "to": {"region": "out", "density": 0.5, "weight": 0.5},
}


class TestNerveExperiment:
    def _setup(self, nerve_cfg: dict | None = None) -> Experiment:
        exp = Experiment()
        exp.setup(_experiment_config(nerve_cfg or _NERVE_CFG_INTEGRATION))
        return exp

    def test_setup_works(self) -> None:
        exp = self._setup()
        assert exp.brain_tensor is not None

    def test_step_runs_without_error(self) -> None:
        exp = self._setup()
        result = exp.step()
        assert result["generation"] == 1

    def test_nerve_from_creates_synapses_in_tissue(self) -> None:
        """Tissue neurons inside the nerve circle get synapses FROM noci."""
        exp = self._setup()
        noci_rs = exp._regions_by_id["noci"]
        on_rs = exp._regions_by_id["tissue"]

        noci_indices = set(range(noci_rs.start, noci_rs.end))
        bt = exp.brain_tensor
        tissue_with_noci = 0
        for neuron_idx in range(on_rs.start, on_rs.end):
            sources = bt.indices_fuente[neuron_idx]
            valid = bt.mascara_valida[neuron_idx]
            for i in range(sources.shape[0]):
                if valid[i] and int(sources[i].item()) in noci_indices:
                    tissue_with_noci += 1
                    break

        assert tissue_with_noci > 0, "No tissue neurons received noci synapses"

    def test_nerve_to_creates_synapses_in_out_region(self) -> None:
        """Output neurons get synapses FROM tissue neurons inside the nerve circle."""
        exp = self._setup()
        on_rs = exp._regions_by_id["tissue"]
        out_rs = exp._regions_by_id["out"]
        tissue_indices = set(range(on_rs.start, on_rs.end))

        bt = exp.brain_tensor
        out_with_tissue = 0
        for neuron_idx in range(out_rs.start, out_rs.end):
            sources = bt.indices_fuente[neuron_idx]
            valid = bt.mascara_valida[neuron_idx]
            for i in range(sources.shape[0]):
                if valid[i] and int(sources[i].item()) in tissue_indices:
                    out_with_tissue += 1
                    break

        assert out_with_tissue > 0, "No output neurons received tissue (nerve) synapses"

    def test_nerve_gradient_center_more_connected(self) -> None:
        """Center tissue neurons receive more noci synapses on average than edge neurons."""
        nerve_cfg = {
            "insertion": {"x": 10, "y": 10},
            "radius": 5,
            "from": {"region": "noci", "density": 1.0, "weight": -0.01},
        }
        center_counts = []
        edge_counts = []
        for seed in range(10):
            random.seed(seed)
            e = Experiment()
            e.setup(_experiment_config(nerve_cfg))
            noci_rs = e._regions_by_id["noci"]
            on_rs = e._regions_by_id["tissue"]
            noci_set = set(range(noci_rs.start, noci_rs.end))
            bt = e.brain_tensor

            def _noci_synapse_count(x: int, y: int) -> int:
                idx = on_rs.start + y * on_rs.width + x
                sources = bt.indices_fuente[idx]
                valid = bt.mascara_valida[idx]
                return sum(1 for i in range(sources.shape[0]) if valid[i] and int(sources[i].item()) in noci_set)

            center_counts.append(_noci_synapse_count(10, 10))
            edge_counts.append(_noci_synapse_count(10, 15))  # on circle edge

        avg_center = sum(center_counts) / len(center_counts)
        avg_edge = sum(edge_counts) / len(edge_counts)
        assert avg_center >= avg_edge, (
            f"Center avg ({avg_center:.2f}) should be >= edge avg ({avg_edge:.2f})"
        )

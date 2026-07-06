"""Tests for delta_weight — per-neuron excitatory/inhibitory total balance.

When a region declares "delta_weight": {"excitatory": e, "inhibitory": i},
every neuron's dendrites of each polarity (regardless of origin — daemon
mask or nerve connection) get rescaled so they sum to e / i respectively.
"""

import random

import pytest
from core.dendrita import Dendrita
from core.neurona import Neurona
from core.region import Region
from core.sinapsis import Sinapsis
from experiments.experiment import Experiment, _apply_delta_weight, _rebalance_dendrite_group


def _dendrita(peso: float) -> Dendrita:
    return Dendrita(sinapsis=[Sinapsis(neurona_entrante=Neurona(id="x"), peso=1.0)], peso=peso)


# ---------------------------------------------------------------------------
# _rebalance_dendrite_group — pure normalization math
# ---------------------------------------------------------------------------

class TestRebalanceDendriteGroup:
    def test_single_dendrite_takes_full_total(self) -> None:
        d = _dendrita(0.3)
        _rebalance_dendrite_group([d], 0.6)
        assert d.peso == pytest.approx(0.6)

    def test_proportional_split_arbitrary_magnitudes(self) -> None:
        """Raw shares 1:2 (José's 10-and-20 example, scaled to a valid dendrite
        range) → 0.2 and 0.4 for a target total of 0.6."""
        d1, d2 = _dendrita(0.1), _dendrita(0.2)
        _rebalance_dendrite_group([d1, d2], 0.6)
        assert d1.peso == pytest.approx(0.2)
        assert d2.peso == pytest.approx(0.4)
        assert d1.peso + d2.peso == pytest.approx(0.6)

    def test_many_dendrites_sum_to_target(self) -> None:
        dends = [_dendrita(-w / 10) for w in (1, 3, 5, 2, 4, 1, 1, 2, 3, 6, 1, 2)]
        _rebalance_dendrite_group(dends, -0.7)
        assert sum(d.peso for d in dends) == pytest.approx(-0.7)

    def test_empty_group_is_noop(self) -> None:
        _rebalance_dendrite_group([], 0.6)  # must not raise

    def test_zero_raw_sum_is_noop(self) -> None:
        d1, d2 = _dendrita(0.0), _dendrita(0.0)
        _rebalance_dendrite_group([d1, d2], 0.6)
        assert d1.peso == 0.0
        assert d2.peso == 0.0

    def test_clamped_to_valid_dendrite_range(self) -> None:
        d = _dendrita(0.1)
        _rebalance_dendrite_group([d], 5.0)
        assert d.peso == 1.0


# ---------------------------------------------------------------------------
# _apply_delta_weight — per-region, per-neuron application
# ---------------------------------------------------------------------------

class TestApplyDeltaWeight:
    def _region_with_neuron(self, dendritas: list[Dendrita]) -> Region:
        region = Region(nombre="tissue")
        n = Neurona(id="n0", dendritas=dendritas)
        region.agregar(n)
        return region

    def test_one_excitatory_one_inhibitory_take_totals_exactly(self) -> None:
        exc, inh = _dendrita(1.0), _dendrita(-1.0)
        region = self._region_with_neuron([exc, inh])
        _apply_delta_weight(region, {"excitatory": 0.6, "inhibitory": -0.7})
        assert exc.peso == pytest.approx(0.6)
        assert inh.peso == pytest.approx(-0.7)

    def test_twelve_inhibitory_dendrites_share_total(self) -> None:
        exc = _dendrita(1.0)
        inhs = [_dendrita(-1.0) for _ in range(12)]
        region = self._region_with_neuron([exc, *inhs])
        _apply_delta_weight(region, {"excitatory": 0.6, "inhibitory": -0.7})
        assert exc.peso == pytest.approx(0.6)
        assert sum(d.peso for d in inhs) == pytest.approx(-0.7)
        # all equal shares since raw magnitudes were equal
        assert all(d.peso == pytest.approx(-0.7 / 12) for d in inhs)

    def test_missing_polarity_left_untouched(self) -> None:
        exc = _dendrita(1.0)
        region = self._region_with_neuron([exc])
        _apply_delta_weight(region, {"excitatory": 0.6, "inhibitory": -0.7})
        assert exc.peso == pytest.approx(0.6)

    def test_only_excitatory_key_set_leaves_inhibitory_alone(self) -> None:
        exc, inh = _dendrita(1.0), _dendrita(-1.0)
        region = self._region_with_neuron([exc, inh])
        _apply_delta_weight(region, {"excitatory": 0.6})
        assert exc.peso == pytest.approx(0.6)
        assert inh.peso == pytest.approx(-1.0)  # untouched


# ---------------------------------------------------------------------------
# Integration: Experiment.setup wires delta_weight end-to-end
# ---------------------------------------------------------------------------

def _tissue_config(delta_weight: dict | None = None, **extra_conns: object) -> dict:
    tissue: dict = {
        "id": "tissue",
        "grid": {"width": 20, "height": 20},
    }
    if delta_weight is not None:
        tissue["delta_weight"] = delta_weight
    return {
        "regions": [tissue, {"id": "input", "grid": {"width": 4, "height": 4}}],
        "connections": [
            {
                "on": "tissue",
                "deamon": {
                    "shape": "square",
                    "excitatory": {"weight": 0.6, "offset": 1, "noise": 0, "weights": [1]},
                    "inhibitory": {"weight": -0.7, "offset": 3, "noise": 0, "sectors": 4, "weights": [1]},
                },
            },
        ],
        **extra_conns,
    }


class TestDeltaWeightIntegration:
    def test_daemon_only_matches_flat_config_exactly(self) -> None:
        """With a single daemon connection, delta_weight must reproduce the
        same per-neuron totals as the pre-existing flat weight override."""
        random.seed(1)
        exp = Experiment()
        exp.setup(_tissue_config(delta_weight={"excitatory": 0.6, "inhibitory": -0.7}))
        neurona = exp.brain.get_neurona("x10y10")
        exc_total = sum(d.peso for d in neurona.dendritas if d.peso > 0)
        inh_total = sum(d.peso for d in neurona.dendritas if d.peso < 0)
        assert exc_total == pytest.approx(0.6)
        assert inh_total == pytest.approx(-0.7)

    def test_daemon_plus_nerve_still_balances_to_totals(self) -> None:
        """The bug this feature fixes: a neuron with daemon + nerve dendrites
        must still land on the configured totals, same as a daemon-only
        neuron elsewhere in the region — no more energetic discontinuity."""
        random.seed(1)
        config = _tissue_config(
            delta_weight={"excitatory": 0.6, "inhibitory": -0.7},
            connections=[
                {
                    "on": "tissue",
                    "deamon": {
                        "shape": "square",
                        "excitatory": {"weight": 0.6, "offset": 1, "noise": 0, "weights": [1]},
                        "inhibitory": {"weight": -0.7, "offset": 3, "noise": 0, "sectors": 4, "weights": [1]},
                    },
                },
                {
                    "on": "tissue",
                    "nerve": {
                        "insertion": {"x": 10, "y": 10},
                        "radius": 5,
                        "from": {"region": "input", "density": 1.0, "weight": 0.2, "gradient": False},
                    },
                },
            ],
        )
        exp = Experiment()
        exp.setup(config)

        inside = exp.brain.get_neurona("x10y10")   # inside the nerve circle
        outside = exp.brain.get_neurona("x1y1")    # outside the nerve circle

        inside_exc = sum(d.peso for d in inside.dendritas if d.peso > 0)
        outside_exc = sum(d.peso for d in outside.dendritas if d.peso > 0)

        assert len(inside.dendritas) > len(outside.dendritas), (
            "sanity check: the nerve must have actually added a dendrite inside the circle"
        )
        assert inside_exc == pytest.approx(0.6)
        assert outside_exc == pytest.approx(0.6)

    def test_without_delta_weight_totals_are_not_forced(self) -> None:
        """Backward compatibility: no delta_weight → raw configured weights stand."""
        random.seed(1)
        config = _tissue_config(
            delta_weight=None,
            connections=[
                {
                    "on": "tissue",
                    "deamon": {
                        "shape": "square",
                        "excitatory": {"weight": 0.6, "offset": 1, "noise": 0, "weights": [1]},
                        "inhibitory": {"weight": -0.7, "offset": 3, "noise": 0, "sectors": 4, "weights": [1]},
                    },
                },
                {
                    "on": "tissue",
                    "nerve": {
                        "insertion": {"x": 10, "y": 10},
                        "radius": 5,
                        "from": {"region": "input", "density": 1.0, "weight": 0.2, "gradient": False},
                    },
                },
            ],
        )
        exp = Experiment()
        exp.setup(config)

        inside = exp.brain.get_neurona("x10y10")
        outside = exp.brain.get_neurona("x1y1")
        inside_exc = sum(d.peso for d in inside.dendritas if d.peso > 0)
        outside_exc = sum(d.peso for d in outside.dendritas if d.peso > 0)

        # inside got an extra 0.2 dendrite from the nerve, on top of daemon's 0.6 —
        # exactly the discontinuity delta_weight is meant to remove.
        assert inside_exc == pytest.approx(0.8)
        assert outside_exc == pytest.approx(0.6)

"""Tests for delta_weight — per-neuron excitatory/inhibitory total balance.

When a region declares "delta_weight": {"excitatory": e, "inhibitory": i},
every neuron's dendrites of each polarity get rebalanced to reproduce exactly
how process_mode="group_avg" combines them at runtime (BrainTensor._gavg):
same-locality dendrites (daemon/mask, intra-region) are AVERAGED together,
distant dendrites (nerve, cross-region) are AVERAGED together separately, and
the two averages are ADDED. So the target is the SUM of the local mean and
the distant mean — not a flat sum across every individual dendrite.
"""

import random

import pytest
from core.dendrita import Dendrita
from core.neurona import Neurona
from core.region import Region
from core.sinapsis import Sinapsis
from experiments.experiment import Experiment, _apply_delta_weight, _rebalance_dendrite_group


def _dendrita(peso: float, source_id: str = "outside") -> Dendrita:
    """A dendrite whose single synapse comes from `source_id`. Locality is
    decided by whether that id is a member of the region passed to
    _rebalance_dendrite_group/_apply_delta_weight."""
    return Dendrita(sinapsis=[Sinapsis(neurona_entrante=Neurona(id=source_id), peso=1.0)], peso=peso)


# ---------------------------------------------------------------------------
# _rebalance_dendrite_group — pure normalization math
# ---------------------------------------------------------------------------

class TestRebalanceDendriteGroup:
    def test_single_dendrite_takes_full_total(self) -> None:
        d = _dendrita(0.3)
        _rebalance_dendrite_group([d], 0.6, local_ids=set())
        assert d.peso == pytest.approx(0.6)

    def test_two_same_locality_dendrites_average_to_target(self) -> None:
        """Both dendrites share one locality bucket (neither id is 'local'),
        so the bucket MEAN — not the raw sum — must hit the target: unequal
        starting shares (1:2) keep that ratio, but land on 0.4/0.8, not 0.2/0.4,
        because averaging two dendrites halves their combined pull on the mean."""
        d1, d2 = _dendrita(0.1), _dendrita(0.2)
        _rebalance_dendrite_group([d1, d2], 0.6, local_ids=set())
        assert d1.peso / d2.peso == pytest.approx(0.1 / 0.2)
        assert (d1.peso + d2.peso) / 2 == pytest.approx(0.6)

    def test_local_and_distant_dendrite_split_proportionally(self) -> None:
        """José's example (10 vs 20, summing to a 0.6 total) is exactly this
        case: one local (daemon) dendrite and one distant (nerve) dendrite,
        each its own single-member bucket — their two means add to target."""
        local_d = _dendrita(0.1, source_id="n0")
        distant_d = _dendrita(0.2, source_id="elsewhere")
        _rebalance_dendrite_group([local_d, distant_d], 0.6, local_ids={"n0"})
        assert local_d.peso == pytest.approx(0.2)
        assert distant_d.peso == pytest.approx(0.4)
        assert local_d.peso + distant_d.peso == pytest.approx(0.6)

    def test_twelve_equal_local_dendrites_land_exactly_on_target(self) -> None:
        """The crown-mask case: 12 equal-weight local inhibitory sectors.
        Their MEAN must equal the target (matching group_avg) — if they
        already averaged to the target, nothing should change."""
        dends = [_dendrita(-0.7, source_id="n0") for _ in range(12)]
        _rebalance_dendrite_group(dends, -0.7, local_ids={"n0"})
        assert all(d.peso == pytest.approx(-0.7) for d in dends)

    def test_empty_group_is_noop(self) -> None:
        _rebalance_dendrite_group([], 0.6, local_ids=set())  # must not raise

    def test_zero_raw_mean_is_noop(self) -> None:
        d1, d2 = _dendrita(0.0), _dendrita(0.0)
        _rebalance_dendrite_group([d1, d2], 0.6, local_ids=set())
        assert d1.peso == 0.0
        assert d2.peso == 0.0

    def test_clamped_to_valid_dendrite_range(self) -> None:
        d = _dendrita(0.1)
        _rebalance_dendrite_group([d], 5.0, local_ids=set())
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

    def test_twelve_equal_inhibitory_dendrites_each_land_on_target(self) -> None:
        """Same-locality (all outside the region → same bucket) equal-weight
        dendrites average to the target: each one ends up AT the target,
        matching what group_avg's mean would compute — not target/12."""
        exc = _dendrita(1.0)
        inhs = [_dendrita(-1.0) for _ in range(12)]
        region = self._region_with_neuron([exc, *inhs])
        _apply_delta_weight(region, {"excitatory": 0.6, "inhibitory": -0.7})
        assert exc.peso == pytest.approx(0.6)
        assert all(d.peso == pytest.approx(-0.7) for d in inhs)

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
        "process_mode": "group_avg",
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
                    "inhibitory": {"weight": -0.7, "offset": 3, "noise": 0, "sectors": 12, "weights": [1]},
                },
            },
        ],
        **extra_conns,
    }


class TestDeltaWeightIntegration:
    def test_daemon_only_with_12_sectors_is_imperceptible(self) -> None:
        """The exact bug José hit: 12 equal inhibitory sectors at -0.7 each.
        With delta_weight={"inhibitory": -0.7} (same number as the flat
        override), every sector must stay at -0.7 — no change at all,
        because their mean already equals the target."""
        random.seed(1)
        exp = Experiment()
        exp.setup(_tissue_config(delta_weight={"excitatory": 0.6, "inhibitory": -0.7}))
        neurona = exp.brain.get_neurona("x10y10")
        exc = [d for d in neurona.dendritas if d.peso > 0]
        inh = [d for d in neurona.dendritas if d.peso < 0]
        assert all(d.peso == pytest.approx(0.6) for d in exc)
        assert all(d.peso == pytest.approx(-0.7) for d in inh)

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
                        "inhibitory": {"weight": -0.7, "offset": 3, "noise": 0, "sectors": 12, "weights": [1]},
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

        def _exc_group_avg_total(neurona, local_ids) -> float:
            exc = [d for d in neurona.dendritas if d.peso > 0]
            local = [d for d in exc if all(s.neurona_entrante.id in local_ids for s in d.sinapsis)]
            distant = [d for d in exc if d not in local]
            local_mean = sum(d.peso for d in local) / len(local) if local else 0.0
            distant_mean = sum(d.peso for d in distant) / len(distant) if distant else 0.0
            return local_mean + distant_mean

        tissue_ids = set(exp.regiones["tissue"].neuronas.keys())
        assert len(inside.dendritas) > len(outside.dendritas), (
            "sanity check: the nerve must have actually added a dendrite inside the circle"
        )
        assert _exc_group_avg_total(inside, tissue_ids) == pytest.approx(0.6)
        assert _exc_group_avg_total(outside, tissue_ids) == pytest.approx(0.6)

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
                        "inhibitory": {"weight": -0.7, "offset": 3, "noise": 0, "sectors": 12, "weights": [1]},
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
        inside_exc = [d.peso for d in inside.dendritas if d.peso > 0]
        outside_exc = [d.peso for d in outside.dendritas if d.peso > 0]

        # inside got an extra 0.2 dendrite from the nerve, on top of daemon's 0.6 —
        # exactly the discontinuity delta_weight is meant to remove.
        assert sorted(inside_exc) == pytest.approx(sorted([0.6, 0.2]))
        assert outside_exc == pytest.approx([0.6])

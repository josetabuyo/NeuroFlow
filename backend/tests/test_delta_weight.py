"""Tests for delta_weight — per-neuron excitatory/inhibitory total balance.

When a region declares "delta_weight": {"positive": e, "negative": i} (both
keys optional, values are always magnitudes >= 0), every neuron's dendrites
of each polarity (bucketed by weight sign) get rebalanced to reproduce
exactly how process_mode="group_avg" combines them at runtime
(BrainTensor._compute_tension): dendrites sharing a declared grupo_id (one
deamon groups[] entry, or one nerve/full connection) are AVERAGED together
first, and those per-group averages are themselves AVERAGED — one vote per
group that's actually present, not weighted by how many raw dendrites it
has. So the target is the AVERAGE of the present groups' means — not a flat
sum/average across every individual dendrite.
"""

import random

import pytest
from core.dendrita import Dendrita
from core.neurona import Neurona
from core.region import Region
from core.sinapsis import Sinapsis
from experiments.experiment import (
    Experiment,
    _apply_delta_weight,
    _convert_delta_weight_entries,
    _rebalance_dendrite_group,
    delta_weight_totals,
)


def _dendrita(peso: float, grupo_id: str = "") -> Dendrita:
    """A dendrite with an optional declared grupo_id — dendrites sharing one
    are the same group for _rebalance_dendrite_group/_apply_delta_weight.
    Empty (default) means "its own singleton group", never merged with
    another empty-grupo_id dendrite."""
    return Dendrita(sinapsis=[Sinapsis(neurona_entrante=Neurona(id="src"), peso=1.0)], peso=peso, grupo_id=grupo_id)


# ---------------------------------------------------------------------------
# _rebalance_dendrite_group — pure normalization math
# ---------------------------------------------------------------------------

class TestRebalanceDendriteGroup:
    def test_single_dendrite_takes_full_total(self) -> None:
        d = _dendrita(0.3)
        _rebalance_dendrite_group([d], 0.6)
        assert d.peso == pytest.approx(0.6)

    def test_two_dendrites_in_same_group_average_to_target(self) -> None:
        """Both dendrites share one declared group, so the group MEAN — not
        the raw sum — must hit the target: unequal starting shares (1:2) keep
        that ratio, but land on 0.4/0.8, not 0.2/0.4, because averaging two
        dendrites halves their combined pull on the mean."""
        d1, d2 = _dendrita(0.1, grupo_id="g0"), _dendrita(0.2, grupo_id="g0")
        _rebalance_dendrite_group([d1, d2], 0.6)
        assert d1.peso / d2.peso == pytest.approx(0.1 / 0.2)
        assert (d1.peso + d2.peso) / 2 == pytest.approx(0.6)

    def test_two_different_groups_vote_equally(self) -> None:
        """Two dendrites in two DIFFERENT declared groups, each its own
        single-member group — their two group-means AVERAGE (one vote per
        group, not per raw dendrite count) to the target, keeping their
        1:2 starting ratio."""
        d1 = _dendrita(0.1, grupo_id="g0")
        d2 = _dendrita(0.2, grupo_id="g1")
        _rebalance_dendrite_group([d1, d2], 0.6)
        assert d1.peso / d2.peso == pytest.approx(0.1 / 0.2)
        assert (d1.peso + d2.peso) / 2 == pytest.approx(0.6)

    def test_twelve_equal_dendrites_in_one_group_land_exactly_on_target(self) -> None:
        """The crown-mask case: 12 equal-weight inhibitory sectors, all one
        declared group. Their MEAN must equal the target (matching
        group_avg) — if they already averaged to the target, nothing changes."""
        dends = [_dendrita(-0.7, grupo_id="g0") for _ in range(12)]
        _rebalance_dendrite_group(dends, -0.7)
        assert all(d.peso == pytest.approx(-0.7) for d in dends)

    def test_empty_group_is_noop(self) -> None:
        _rebalance_dendrite_group([], 0.6)  # must not raise

    def test_zero_raw_mean_is_noop(self) -> None:
        d1, d2 = _dendrita(0.0, grupo_id="g0"), _dendrita(0.0, grupo_id="g1")
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
        _apply_delta_weight(region, {"positive": 0.6, "negative": 0.7})
        assert exc.peso == pytest.approx(0.6)
        assert inh.peso == pytest.approx(-0.7)

    def test_twelve_equal_inhibitory_dendrites_each_land_on_target(self) -> None:
        """Same-locality (all outside the region → same bucket) equal-weight
        dendrites average to the target: each one ends up AT the target,
        matching what group_avg's mean would compute — not target/12."""
        exc = _dendrita(1.0)
        inhs = [_dendrita(-1.0) for _ in range(12)]
        region = self._region_with_neuron([exc, *inhs])
        _apply_delta_weight(region, {"positive": 0.6, "negative": 0.7})
        assert exc.peso == pytest.approx(0.6)
        assert all(d.peso == pytest.approx(-0.7) for d in inhs)

    def test_missing_polarity_left_untouched(self) -> None:
        exc = _dendrita(1.0)
        region = self._region_with_neuron([exc])
        _apply_delta_weight(region, {"positive": 0.6, "negative": 0.7})
        assert exc.peso == pytest.approx(0.6)

    def test_only_positive_key_set_leaves_negative_alone(self) -> None:
        exc, inh = _dendrita(1.0), _dendrita(-1.0)
        region = self._region_with_neuron([exc, inh])
        _apply_delta_weight(region, {"positive": 0.6})
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
                    "groups": [
                        {"id": "first_ring", "weight": 0.6, "noise": 0, "weights": [1]},
                        {"id": "second_ring", "weight": -0.7, "gap": 1, "noise": 0, "sectors": 12, "weights": [1]},
                    ],
                },
            },
        ],
        **extra_conns,
    }


class TestDeltaWeightIntegration:
    def test_daemon_only_with_12_sectors_is_imperceptible(self) -> None:
        """The exact bug José hit: 12 equal inhibitory sectors at -0.7 each.
        With delta_weight targeting -0.7 (same number as the flat override),
        every sector must stay at -0.7 — no change at all, because their
        mean already equals the target."""
        random.seed(1)
        exp = Experiment()
        exp.setup(_tissue_config(delta_weight={"positive": 0.6, "negative": 0.7}))
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
            delta_weight={"positive": 0.6, "negative": 0.7},
            connections=[
                {
                    "on": "tissue",
                    "deamon": {
                        "shape": "square",
                        "groups": [
                            {"id": "first_ring", "weight": 0.6, "offset": 1, "noise": 0, "weights": [1]},
                            {"id": "second_ring", "weight": -0.7, "offset": 3, "noise": 0, "sectors": 12, "weights": [1]},
                        ],
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

        def _exc_group_avg_total(neurona) -> float:
            exc = [d for d in neurona.dendritas if d.peso > 0]
            by_grupo: dict[str, list] = {}
            for d in exc:
                by_grupo.setdefault(d.grupo_id or f"__singleton__{id(d)}", []).append(d)
            grupo_means = [sum(d.peso for d in ds) / len(ds) for ds in by_grupo.values()]
            return sum(grupo_means) / len(grupo_means) if grupo_means else 0.0

        assert len(inside.dendritas) > len(outside.dendritas), (
            "sanity check: the nerve must have actually added a dendrite inside the circle"
        )
        assert _exc_group_avg_total(inside) == pytest.approx(0.6)
        assert _exc_group_avg_total(outside) == pytest.approx(0.6)

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
                        "groups": [
                            {"id": "first_ring", "weight": 0.6, "offset": 1, "noise": 0, "weights": [1]},
                            {"id": "second_ring", "weight": -0.7, "offset": 3, "noise": 0, "sectors": 12, "weights": [1]},
                        ],
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


# ---------------------------------------------------------------------------
# delta_weight_totals — key is the sole source of sign
# ---------------------------------------------------------------------------

class TestDeltaWeightTotals:
    def test_both_keys_present(self) -> None:
        exc, inh = delta_weight_totals({"positive": 0.6, "negative": 0.7})
        assert exc == pytest.approx(0.6)
        assert inh == pytest.approx(-0.7)

    def test_only_positive_present_leaves_negative_none(self) -> None:
        exc, inh = delta_weight_totals({"positive": 0.6})
        assert exc == pytest.approx(0.6)
        assert inh is None

    def test_only_negative_present_leaves_positive_none(self) -> None:
        exc, inh = delta_weight_totals({"negative": 0.7})
        assert exc is None
        assert inh == pytest.approx(-0.7)

    def test_neither_key_present_returns_both_none(self) -> None:
        exc, inh = delta_weight_totals({})
        assert exc is None
        assert inh is None

    def test_value_is_always_treated_as_magnitude(self) -> None:
        """Even if the "negative" value is (incorrectly) given as
        already-negative, the result must still be exactly one negative
        value — the key, not the value's own sign, decides."""
        exc, inh = delta_weight_totals({"negative": -0.7})
        assert exc is None
        assert inh == pytest.approx(-0.7)


# ---------------------------------------------------------------------------
# _convert_delta_weight_entries — legacy shapes -> {"positive"?, "negative"?}
# ---------------------------------------------------------------------------

class TestConvertDeltaWeightEntries:
    def test_legacy_signed_dict_shape_converts(self) -> None:
        result = _convert_delta_weight_entries({"excitatory": 0.6, "inhibitory": -0.7})
        assert result == {"positive": 0.6, "negative": 0.7}

    def test_intermediate_id_list_shape_converts(self) -> None:
        result = _convert_delta_weight_entries(
            [{"id": "excitatory", "weight": 0.6}, {"id": "inhibitory", "weight": -0.7}]
        )
        assert result == {"positive": 0.6, "negative": 0.7}

    def test_intermediate_polarity_list_shape_converts(self) -> None:
        result = _convert_delta_weight_entries(
            [{"polarity": "positive", "weight": 0.6}, {"polarity": "negative", "weight": 0.7}]
        )
        assert result == {"positive": 0.6, "negative": 0.7}

    def test_already_current_dict_shape_is_noop(self) -> None:
        already = {"positive": 0.6, "negative": 0.7}
        assert _convert_delta_weight_entries(already) is already

    def test_none_passes_through(self) -> None:
        assert _convert_delta_weight_entries(None) is None

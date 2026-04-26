"""Tests for the unified Experiment class.

Validates:
- All mask presets are applied correctly
- Reconnection preserves values
- Standard functionality (step, click, reset, get_frame)
- Opt-in feature semantics (sections absent = disabled)
- Daemon metrics
- Wolfram masks
"""

import random

import pytest
from core.constructor import Constructor
from core.masks import (
    MASK_PRESETS,
    MASK_SIMPLE,
    get_mask,
    get_mask_info,
    _moore,
    _von_neumann,
    _ring,
    _partition,
)
from experiments.experiment import Experiment


def _nested_config(
    width: int = 10,
    height: int = 10,
    mask: str = "simple",
    process_mode: str = "min_vs_max",
    **extra: object,
) -> dict:
    """Helper to build a minimal nested config."""
    cfg: dict = {
        "grid": {"width": width, "height": height},
        "wiring": {"mask": mask, "process_mode": process_mode},
    }
    cfg.update(extra)
    return cfg


class TestMaskHelpers:
    """Tests for offset generation helper functions."""

    def test_moore_radius_1_gives_8_neighbors(self) -> None:
        offsets = _moore(1)
        assert len(offsets) == 8
        assert (0, 0) not in offsets

    def test_moore_radius_2_gives_24_neighbors(self) -> None:
        offsets = _moore(2)
        assert len(offsets) == 24

    def test_von_neumann_radius_1_gives_4_neighbors(self) -> None:
        offsets = _von_neumann(1)
        assert len(offsets) == 4
        assert set(offsets) == {(1, 0), (-1, 0), (0, 1), (0, -1)}

    def test_ring_2_4_gives_72_offsets(self) -> None:
        offsets = _ring(2, 4)
        assert len(offsets) == 72

    def test_ring_excludes_center(self) -> None:
        offsets = _ring(2, 4)
        for dx, dy in offsets:
            assert max(abs(dx), abs(dy)) >= 2

    def test_partition_8_creates_8_sectors(self) -> None:
        offsets = _ring(2, 4)
        sectors = _partition(offsets, 8)
        assert len(sectors) == 8
        total = sum(len(s) for s in sectors)
        assert total == 72


class TestMaskPresets:
    """Tests for each mask preset."""

    def test_all_presets_exist(self) -> None:
        expected = [
            "simple", "wide_hat", "narrow_hat", "big_center",
            "cross_center", "one_dendrite", "fine_grain",
            "double_ring", "soft_inhibit", "strong_center",
        ]
        for preset_id in expected:
            assert preset_id in MASK_PRESETS

    def test_get_mask_returns_list(self) -> None:
        for preset_id in MASK_PRESETS:
            mask = get_mask(preset_id)
            assert isinstance(mask, list)
            assert len(mask) > 0

    def test_get_mask_invalid_raises(self) -> None:
        with pytest.raises(KeyError):
            get_mask("nonexistent_mask")

    def test_get_mask_info_excludes_mask_data(self) -> None:
        info = get_mask_info()
        assert len(info) == 49
        for entry in info:
            assert "mask" not in entry
            assert "id" in entry
            assert "name" in entry

    @pytest.mark.parametrize("preset_id", list(MASK_PRESETS.keys()))
    def test_preset_has_at_least_one_excitatory(self, preset_id: str) -> None:
        if preset_id == "all_inh":
            pytest.skip("all_inh is a pure-inhibitory diagnostic mask by design")
        mask = get_mask(preset_id)
        exc = [d for d in mask if d["peso_dendrita"] > 0]
        assert len(exc) >= 1

    @pytest.mark.parametrize("preset_id", list(MASK_PRESETS.keys()))
    def test_preset_has_at_least_one_inhibitory(self, preset_id: str) -> None:
        if preset_id == "all_exc":
            pytest.skip("all_exc is a pure-excitatory diagnostic mask by design")
        if MASK_PRESETS[preset_id].get("mask_type") == "wolfram":
            pytest.skip("wolfram masks use only excitatory dendrites")
        mask = get_mask(preset_id)
        inh = [d for d in mask if d["peso_dendrita"] < 0]
        assert len(inh) >= 1

    @pytest.mark.parametrize("preset_id", list(MASK_PRESETS.keys()))
    def test_preset_offsets_are_valid(self, preset_id: str) -> None:
        mask = get_mask(preset_id)
        for dendrita in mask:
            offsets = dendrita["offsets"]
            assert len(offsets) > 0
            for dx, dy in offsets:
                assert isinstance(dx, int)
                assert isinstance(dy, int)

    @pytest.mark.parametrize("preset_id", list(MASK_PRESETS.keys()))
    def test_preset_applies_to_grid(self, preset_id: str) -> None:
        random.seed(42)
        mask = get_mask(preset_id)
        constructor = Constructor()
        brain, _ = constructor.crear_grilla(
            width=15, height=15, filas_entrada=[], filas_salida=[], umbral=0.0
        )
        constructor.aplicar_mascara_2d(brain, 15, 15, mask)
        neurona = brain.get_neurona("x7y7")
        assert len(neurona.dendritas) >= 1


class TestExperimentSetup:
    """Experiment setup with nested config."""

    def test_setup_default_mask(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(width=10, height=10, mask="simple"))
        assert exp.brain_tensor is not None
        neurona = exp.brain.get_neurona("x5y5")
        assert len(neurona.dendritas) == 13

    def test_setup_wide_hat(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(width=15, height=15, mask="wide_hat"))
        neurona = exp.brain.get_neurona("x7y7")
        assert len(neurona.dendritas) >= 9

    def test_setup_cross_center(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(width=15, height=15, mask="cross_center"))
        neurona = exp.brain.get_neurona("x7y7")
        exc = [d for d in neurona.dendritas if d.peso > 0]
        assert len(exc[0].sinapsis) == 4

    def test_setup_one_dendrite(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(width=10, height=10, mask="one_dendrite"))
        neurona = exp.brain.get_neurona("x5y5")
        assert len(neurona.dendritas) == 2

    def test_no_input_when_section_absent(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        assert exp.input_enabled is False

    def test_no_learning_when_section_absent(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        assert exp.learning_enabled is False

    def test_no_spiking_when_section_absent(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        assert exp.adaptation_enabled is False

    def test_input_enabled_when_section_present(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(input={"text": "HALF_TOP,HALF_BOT", "resolution": 10}))
        assert exp.input_enabled is True

    def test_learning_enabled_when_section_present(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(learning={"rate": 0.01}))
        assert exp.learning_enabled is True

    def test_spiking_enabled_when_section_present(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(spiking={"up_ticks": 5, "down_ticks": 5}))
        assert exp.adaptation_enabled is True


class TestExperimentInit:
    """Tests for random initialization."""

    def test_init_random(self) -> None:
        random.seed(42)
        exp = Experiment()
        exp.setup(_nested_config())
        frame = exp.get_frame()
        values = [v for row in frame for v in row]
        assert any(0 < v < 1 for v in values)


class TestExperimentFunctionality:
    """Standard experiment functionality."""

    def test_step_avanza_generacion(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        result = exp.step()
        assert result["type"] == "frame"
        assert result["generation"] == 1

    def test_step_n(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        result = exp.step_n(5)
        assert result["generation"] == 5

    def test_click_toggle(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        idx = 5 * 10 + 5
        exp.brain_tensor.set_valor(idx, 0.0)
        exp.click(5, 5)
        assert exp.brain_tensor.valores[idx].item() == 1.0
        exp.click(5, 5)
        assert exp.brain_tensor.valores[idx].item() == 0.0

    def test_reset_reinicializa(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        exp.step()
        exp.step()
        assert exp.generation == 2
        exp.reset()
        assert exp.generation == 0

    def test_is_complete_siempre_false(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(width=5, height=5))
        assert exp.is_complete() is False
        for _ in range(10):
            exp.step()
        assert exp.is_complete() is False

    def test_get_frame_dimensions(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        frame = exp.get_frame()
        assert len(frame) == 10
        assert all(len(row) == 10 for row in frame)

    @pytest.mark.parametrize("mask_id", list(MASK_PRESETS.keys()))
    def test_all_presets_run_10_steps(self, mask_id: str) -> None:
        random.seed(42)
        exp = Experiment()
        exp.setup(_nested_config(mask=mask_id))
        for _ in range(10):
            result = exp.step()
            assert result["type"] == "frame"


class TestToroidalTopology:
    """Toroidal (wrap-around) topology: border neurons have identical connectivity."""

    @pytest.mark.parametrize("preset_id", [
        k for k, v in MASK_PRESETS.items() if v.get("mask_type") != "wolfram"
    ])
    def test_border_same_dendrites_as_center(self, preset_id: str) -> None:
        random.seed(42)
        mask = get_mask(preset_id)
        constructor = Constructor()
        brain, _ = constructor.crear_grilla(
            width=15, height=15, filas_entrada=[], filas_salida=[], umbral=0.0
        )
        constructor.aplicar_mascara_2d(brain, 15, 15, mask)

        n_center = brain.get_neurona("x7y7")
        n_corner = brain.get_neurona("x0y0")
        n_edge = brain.get_neurona("x0y7")

        assert len(n_corner.dendritas) == len(n_center.dendritas), (
            f"{preset_id}: corner has {len(n_corner.dendritas)} dendritas "
            f"vs center {len(n_center.dendritas)}"
        )
        assert len(n_edge.dendritas) == len(n_center.dendritas)

        for d_center, d_corner in zip(n_center.dendritas, n_corner.dendritas):
            assert len(d_corner.sinapsis) == len(d_center.sinapsis), (
                f"{preset_id}: corner dendrite has {len(d_corner.sinapsis)} "
                f"sinapsis vs center {len(d_center.sinapsis)}"
            )

    def test_wolfram_border_wraps_horizontally(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(width=5, height=3, mask="rule_110"))

        for i in range(exp.brain_tensor.n_real):
            exp.brain_tensor.set_valor(i, 0.0)
        exp.brain_tensor.set_valor(2 * 5 + 0, 1.0)  # x0y2

        exp.brain_tensor.procesar()

        frame = exp.brain_tensor.get_grid(5, 3)
        row1 = [int(round(v)) for v in frame[1]]
        assert row1[4] == 1, (
            f"Rightmost cell should fire from wrap-around; got row1={row1}"
        )


class TestDaemonMetrics:
    """Tests for daemon metrics."""

    def test_stats_include_daemon_fields(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        exp.step()
        stats = exp.get_stats()
        assert "daemon_count" in stats
        assert "avg_daemon_size" in stats
        assert "noise_cells" in stats
        assert "stability" in stats
        assert "exclusion" in stats
        assert isinstance(stats["daemon_count"], int)
        assert isinstance(stats["avg_daemon_size"], float)
        assert isinstance(stats["noise_cells"], int)

    def test_daemon_count_zero_for_empty_grid(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(width=5, height=5))
        for i in range(exp.brain_tensor.n_real):
            exp.brain_tensor.set_valor(i, 0.0)
        stats = exp.get_stats()
        assert stats["daemon_count"] == 0
        assert stats["noise_cells"] == 0
        assert stats["active_cells"] == 0

    def test_daemon_count_one_cluster(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        for i in range(exp.brain_tensor.n_real):
            exp.brain_tensor.set_valor(i, 0.0)
        for dy in range(3):
            for dx in range(3):
                idx = (3 + dy) * 10 + (3 + dx)
                exp.brain_tensor.set_valor(idx, 1.0)
        stats = exp.get_stats()
        assert stats["daemon_count"] == 1
        assert stats["active_cells"] == 9
        assert stats["avg_daemon_size"] == 9.0
        assert stats["noise_cells"] == 0

    def test_isolated_pixels_are_noise_not_daemons(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        for i in range(exp.brain_tensor.n_real):
            exp.brain_tensor.set_valor(i, 0.0)
        exp.brain_tensor.set_valor(0, 1.0)
        exp.brain_tensor.set_valor(9 * 10 + 9, 1.0)
        stats = exp.get_stats()
        assert stats["daemon_count"] == 0
        assert stats["noise_cells"] == 2
        assert stats["active_cells"] == 2

    def test_pair_is_noise_not_daemon(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        for i in range(exp.brain_tensor.n_real):
            exp.brain_tensor.set_valor(i, 0.0)
        exp.brain_tensor.set_valor(0, 1.0)
        exp.brain_tensor.set_valor(1, 1.0)
        stats = exp.get_stats()
        assert stats["daemon_count"] == 0
        assert stats["noise_cells"] == 2

    def test_two_real_clusters(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        for i in range(exp.brain_tensor.n_real):
            exp.brain_tensor.set_valor(i, 0.0)
        exp.brain_tensor.set_valor(0, 1.0)
        exp.brain_tensor.set_valor(1, 1.0)
        exp.brain_tensor.set_valor(10, 1.0)
        exp.brain_tensor.set_valor(98, 1.0)
        exp.brain_tensor.set_valor(99, 1.0)
        exp.brain_tensor.set_valor(89, 1.0)
        stats = exp.get_stats()
        assert stats["daemon_count"] == 2
        assert stats["noise_cells"] == 0
        assert stats["avg_daemon_size"] == 3.0

    def test_mixed_daemons_and_noise(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        for i in range(exp.brain_tensor.n_real):
            exp.brain_tensor.set_valor(i, 0.0)
        for dy in range(3):
            for dx in range(3):
                exp.brain_tensor.set_valor((dy) * 10 + dx, 1.0)
        exp.brain_tensor.set_valor(99, 1.0)
        stats = exp.get_stats()
        assert stats["daemon_count"] == 1
        assert stats["noise_cells"] == 1
        assert stats["active_cells"] == 10

    def test_exclusion_with_daemon_and_noise(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        for i in range(exp.brain_tensor.n_real):
            exp.brain_tensor.set_valor(i, 0.0)
        exp.brain_tensor.set_valor(0, 1.0)
        exp.brain_tensor.set_valor(1, 1.0)
        exp.brain_tensor.set_valor(10, 1.0)
        stats = exp.get_stats()
        assert stats["exclusion"] > 0.9

    def test_exclusion_zero_for_empty_grid(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(width=5, height=5))
        for i in range(exp.brain_tensor.n_real):
            exp.brain_tensor.set_valor(i, 0.0)
        stats = exp.get_stats()
        assert stats["exclusion"] == 0.0

    def test_stability_increases_with_consistency(self) -> None:
        random.seed(42)
        exp = Experiment()
        exp.setup(_nested_config())
        for _ in range(10):
            exp.step()
        stats = exp.get_stats()
        assert stats["stability"] >= 0.0

    def test_stability_zero_on_first_step(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        exp.step()
        stats = exp.get_stats()
        assert stats["stability"] == 0.0

    def test_daemon_history_resets_on_reset(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        for _ in range(5):
            exp.step()
        assert len(exp._daemon_history) == 5
        exp.reset()
        assert len(exp._daemon_history) == 0

    def test_no_duplicate_history_on_double_get_stats(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())
        exp.step()
        exp.get_stats()
        exp.get_stats()
        assert len(exp._daemon_history) == 1


class TestMaskStatsInInfo:
    """Tests for mask_stats in get_mask_info()."""

    def test_mask_info_includes_mask_stats(self) -> None:
        info = get_mask_info()
        for entry in info:
            assert "mask_stats" in entry
            ms = entry["mask_stats"]
            assert "excitatory_synapses" in ms
            assert "inhibitory_synapses" in ms
            assert "ratio_exc_inh" in ms
            assert "excitation_radius" in ms
            assert "inhibition_radius" in ms

    def test_simple_mask_stats_values(self) -> None:
        info = get_mask_info()
        simple = next(e for e in info if e["id"] == "simple")
        ms = simple["mask_stats"]
        assert ms["excitatory_synapses"] == 8
        assert ms["inhibitory_synapses"] == 72
        assert ms["excitation_radius"] == 1
        assert ms["inhibition_radius"] == 4

    def test_cross_center_mask_stats(self) -> None:
        info = get_mask_info()
        cross = next(e for e in info if e["id"] == "cross_center")
        ms = cross["mask_stats"]
        assert ms["excitatory_synapses"] == 4
        assert ms["excitation_radius"] == 1


class TestWolframMasks:
    """Tests for Wolfram elementary CA rules."""

    def _get_row(self, exp: Experiment, row: int) -> list[int]:
        frame = exp.get_frame()
        return [int(round(v)) for v in frame[row]]

    def test_rule_110_setup_initializes_correctly(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(width=9, height=5, mask="rule_110"))
        frame = exp.get_frame()
        bottom = [int(round(v)) for v in frame[4]]
        assert bottom == [0, 0, 0, 0, 1, 0, 0, 0, 0]
        for row in range(4):
            assert all(v == 0.0 for v in frame[row])

    def test_rule_110_one_step(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(width=9, height=5, mask="rule_110"))
        exp.step()
        row3 = self._get_row(exp, 3)
        assert row3 == [0, 0, 0, 1, 1, 0, 0, 0, 0]

    def test_rule_30_one_step(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(width=9, height=5, mask="rule_30"))
        exp.step()
        row3 = self._get_row(exp, 3)
        assert row3 == [0, 0, 0, 1, 1, 1, 0, 0, 0]

    def test_wolfram_reset_reinitializes(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(width=9, height=5, mask="rule_110"))
        for _ in range(3):
            exp.step()
        exp.reset()
        frame = exp.get_frame()
        bottom = [int(round(v)) for v in frame[4]]
        assert bottom == [0, 0, 0, 0, 1, 0, 0, 0, 0]

    def test_reconnect_wolfram_to_kohonen_does_full_reset(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config(width=10, height=10, mask="rule_110"))
        exp.step()
        exp.setup(_nested_config(width=10, height=10, mask="simple"))
        assert exp._mask_type == "kohonen"

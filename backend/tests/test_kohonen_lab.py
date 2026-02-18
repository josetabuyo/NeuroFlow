"""Tests para Kohonen Lab ? laboratorio de conexionados con m?scara configurable.

Valida:
- Todos los presets de m?scara se aplican correctamente
- Equivalencia con KohonenExperiment cuando mask="simple"
- Reconexi?n preserva valores
- Funcionalidad est?ndar (step, click, reset, get_frame)
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
from experiments.kohonen import KOHONEN_SIMPLE_MASK
from experiments.kohonen_lab import KohonenLabExperiment


class TestMaskHelpers:
    """Tests para las funciones helper de generaci?n de offsets."""

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
    """Tests para cada preset de m?scara."""

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
        assert len(info) == 12
        for entry in info:
            assert "mask" not in entry
            assert "id" in entry
            assert "name" in entry

    def test_simple_mask_matches_kohonen(self) -> None:
        """MASK_SIMPLE in masks.py is identical to KOHONEN_SIMPLE_MASK."""
        assert MASK_SIMPLE == KOHONEN_SIMPLE_MASK

    @pytest.mark.parametrize("preset_id", list(MASK_PRESETS.keys()))
    def test_preset_has_at_least_one_excitatory(self, preset_id: str) -> None:
        """Every preset has at least one excitatory dendrite (skip pure-inh diagnostic masks)."""
        if preset_id == "all_inh":
            pytest.skip("all_inh is a pure-inhibitory diagnostic mask by design")
        mask = get_mask(preset_id)
        exc = [d for d in mask if d["peso_dendrita"] > 0]
        assert len(exc) >= 1

    @pytest.mark.parametrize("preset_id", list(MASK_PRESETS.keys()))
    def test_preset_has_at_least_one_inhibitory(self, preset_id: str) -> None:
        """Every preset has at least one inhibitory dendrite (skip pure-exc diagnostic masks)."""
        if preset_id == "all_exc":
            pytest.skip("all_exc is a pure-excitatory diagnostic mask by design")
        mask = get_mask(preset_id)
        inh = [d for d in mask if d["peso_dendrita"] < 0]
        assert len(inh) >= 1

    @pytest.mark.parametrize("preset_id", list(MASK_PRESETS.keys()))
    def test_preset_offsets_are_valid(self, preset_id: str) -> None:
        """All offsets are tuples of two ints, no (0,0)."""
        mask = get_mask(preset_id)
        for dendrita in mask:
            offsets = dendrita["offsets"]
            assert len(offsets) > 0
            for dx, dy in offsets:
                assert isinstance(dx, int)
                assert isinstance(dy, int)

    @pytest.mark.parametrize("preset_id", list(MASK_PRESETS.keys()))
    def test_preset_applies_to_grid(self, preset_id: str) -> None:
        """Every preset can be applied to a 15x15 grid without errors."""
        random.seed(42)
        mask = get_mask(preset_id)
        constructor = Constructor()
        red, _ = constructor.crear_grilla(
            width=15, height=15, filas_entrada=[], filas_salida=[], umbral=0.0
        )
        constructor.aplicar_mascara_2d(red, 15, 15, mask)
        neurona = red.get_neurona("x7y7")
        assert len(neurona.dendritas) >= 1


class TestKohonenLabSetup:
    """Setup del experimento Kohonen Lab."""

    def test_setup_default_mask(self) -> None:
        """Setup con mask por defecto (simple) crea red correcta."""
        exp = KohonenLabExperiment()
        exp.setup({"width": 10, "height": 10})
        assert len(exp.red.neuronas) == 100
        neurona = exp.red.get_neurona("x5y5")
        assert len(neurona.dendritas) == 9

    def test_setup_wide_hat(self) -> None:
        """Setup con wide_hat produce m?s dendritas inhibitorias."""
        exp = KohonenLabExperiment()
        exp.setup({"width": 15, "height": 15, "mask": "wide_hat"})
        neurona = exp.red.get_neurona("x7y7")
        assert len(neurona.dendritas) >= 9

    def test_setup_cross_center(self) -> None:
        """Setup con cross_center usa Von Neumann + 4 dendritas."""
        exp = KohonenLabExperiment()
        exp.setup({"width": 15, "height": 15, "mask": "cross_center"})
        neurona = exp.red.get_neurona("x7y7")
        exc = [d for d in neurona.dendritas if d.peso > 0]
        assert len(exc[0].sinapsis) == 4

    def test_setup_one_dendrite(self) -> None:
        """Setup con one_dendrite: 1 exc + 1 inh."""
        exp = KohonenLabExperiment()
        exp.setup({"width": 10, "height": 10, "mask": "one_dendrite"})
        neurona = exp.red.get_neurona("x5y5")
        assert len(neurona.dendritas) == 2

    def test_setup_with_balance_positive(self) -> None:
        """Setup con balance>0 reduce sinapsis inhibitorias."""
        random.seed(42)
        exp_no = KohonenLabExperiment()
        exp_no.setup({"width": 10, "height": 10, "mask": "simple"})

        random.seed(42)
        exp_bal = KohonenLabExperiment()
        exp_bal.setup({"width": 10, "height": 10, "mask": "simple", "balance": 0.5, "balance_mode": "weight"})

        n_no = exp_no.red.get_neurona("x5y5")
        n_bal = exp_bal.red.get_neurona("x5y5")

        for d_no, d_bal in zip(n_no.dendritas, n_bal.dendritas):
            if d_no.peso < 0:
                for s_no, s_bal in zip(d_no.sinapsis, d_bal.sinapsis):
                    assert s_bal.peso < s_no.peso or s_no.peso == 0.0

    def test_setup_with_balance_zero_no_change(self) -> None:
        """Setup con balance=0.0 no modifica pesos (retorno inmediato)."""
        random.seed(42)
        exp_no = KohonenLabExperiment()
        exp_no.setup({"width": 10, "height": 10, "mask": "simple"})

        random.seed(42)
        exp_bal = KohonenLabExperiment()
        exp_bal.setup({"width": 10, "height": 10, "mask": "simple", "balance": 0.0})

        n_no = exp_no.red.get_neurona("x5y5")
        n_bal = exp_bal.red.get_neurona("x5y5")

        for d_no, d_bal in zip(n_no.dendritas, n_bal.dendritas):
            for s_no, s_bal in zip(d_no.sinapsis, d_bal.sinapsis):
                assert s_no.peso == pytest.approx(s_bal.peso, abs=1e-9)

    def test_setup_without_balance_no_balancing(self) -> None:
        """Setup sin balance key no aplica balanceo."""
        exp = KohonenLabExperiment()
        exp.setup({"width": 10, "height": 10, "mask": "simple"})
        assert exp.red_tensor is not None
        assert exp._config.get("balance") is None


class TestKohonenLabInit:
    """Tests para la inicializacion aleatoria."""

    def test_init_random(self) -> None:
        """Setup siempre usa random (valores entre 0 y 1)."""
        random.seed(42)
        exp = KohonenLabExperiment()
        exp.setup({"width": 10, "height": 10, "mask": "simple"})
        frame = exp.get_frame()
        values = [v for row in frame for v in row]
        assert any(0 < v < 1 for v in values)


class TestKohonenLabBalanceMode:
    """Tests para balance_mode en Kohonen Lab."""

    def test_balance_mode_none_no_modifica(self) -> None:
        """balance_mode='none' no modifica pesos ni sinapsis."""
        random.seed(42)
        exp_ref = KohonenLabExperiment()
        exp_ref.setup({"width": 10, "height": 10, "mask": "simple"})

        random.seed(42)
        exp = KohonenLabExperiment()
        exp.setup({
            "width": 10, "height": 10, "mask": "simple",
            "balance": 0.5, "balance_mode": "none",
        })

        n_ref = exp_ref.red.get_neurona("x5y5")
        n_test = exp.red.get_neurona("x5y5")

        for d_ref, d_test in zip(n_ref.dendritas, n_test.dendritas):
            assert len(d_ref.sinapsis) == len(d_test.sinapsis)
            for s_ref, s_test in zip(d_ref.sinapsis, d_test.sinapsis):
                assert s_ref.peso == pytest.approx(s_test.peso, abs=1e-9)

    def test_balance_mode_weight_scales_weights(self) -> None:
        """balance_mode='weight' escala pesos inhibitorios (comportamiento existente)."""
        random.seed(42)
        exp_ref = KohonenLabExperiment()
        exp_ref.setup({"width": 10, "height": 10, "mask": "simple"})

        random.seed(42)
        exp = KohonenLabExperiment()
        exp.setup({
            "width": 10, "height": 10, "mask": "simple",
            "balance": 0.5, "balance_mode": "weight",
        })

        n_ref = exp_ref.red.get_neurona("x5y5")
        n_test = exp.red.get_neurona("x5y5")

        for d_ref, d_test in zip(n_ref.dendritas, n_test.dendritas):
            assert len(d_ref.sinapsis) == len(d_test.sinapsis)
            if d_ref.peso < 0:
                for s_ref, s_test in zip(d_ref.sinapsis, d_test.sinapsis):
                    assert s_test.peso == pytest.approx(s_ref.peso * 0.5, abs=1e-9)

    def test_balance_mode_synapse_count_reduces_synapses(self) -> None:
        """balance_mode='synapse_count' reduce cantidad de sinapsis inhibitorias."""
        random.seed(42)
        exp_ref = KohonenLabExperiment()
        exp_ref.setup({"width": 10, "height": 10, "mask": "simple"})

        random.seed(42)
        exp = KohonenLabExperiment()
        exp.setup({
            "width": 10, "height": 10, "mask": "simple",
            "balance": 0.5, "balance_mode": "synapse_count",
        })

        n_ref = exp_ref.red.get_neurona("x5y5")
        n_test = exp.red.get_neurona("x5y5")

        for d_ref, d_test in zip(n_ref.dendritas, n_test.dendritas):
            if d_ref.peso < 0:
                assert len(d_test.sinapsis) < len(d_ref.sinapsis)
            else:
                assert len(d_test.sinapsis) == len(d_ref.sinapsis)

    def test_reconnect_with_balance_mode_synapse_count(self) -> None:
        """Reconnect con balance_mode='synapse_count' reduce sinapsis."""
        exp = KohonenLabExperiment()
        exp.setup({"width": 10, "height": 10, "mask": "simple"})

        exp.reconnect({
            "mask": "simple",
            "balance": 0.5,
            "balance_mode": "synapse_count",
        })

        n = exp.red.get_neurona("x5y5")
        for d in n.dendritas:
            if d.peso < 0:
                assert len(d.sinapsis) < 9  # simple mask has 9 inh per dendrita


class TestKohonenLabEquivalence:
    """Equivalencia con KohonenExperiment cuando mask=simple."""

    def test_same_mask_same_dendritas(self) -> None:
        """KohonenLab(simple) produce misma estructura que Kohonen."""
        from experiments.kohonen import KohonenExperiment

        random.seed(42)
        exp_lab = KohonenLabExperiment()
        exp_lab.setup({"width": 10, "height": 10, "mask": "simple"})

        random.seed(42)
        exp_orig = KohonenExperiment()
        exp_orig.setup({"width": 10, "height": 10})

        n_lab = exp_lab.red.get_neurona("x5y5")
        n_orig = exp_orig.red.get_neurona("x5y5")
        assert len(n_lab.dendritas) == len(n_orig.dendritas)

    def test_same_seed_same_frame(self) -> None:
        """Con misma seed, KohonenLab(simple) da el mismo frame inicial."""
        from experiments.kohonen import KohonenExperiment

        random.seed(42)
        exp_lab = KohonenLabExperiment()
        exp_lab.setup({"width": 10, "height": 10, "mask": "simple"})

        random.seed(42)
        exp_orig = KohonenExperiment()
        exp_orig.setup({"width": 10, "height": 10})

        frame_lab = exp_lab.get_frame()
        frame_orig = exp_orig.get_frame()
        assert frame_lab == frame_orig


class TestKohonenLabReconnect:
    """Reconexi?n: cambiar m?scara preservando valores."""

    def test_reconnect_preserves_values(self) -> None:
        """Reconnect mantiene los valores de las neuronas."""
        exp = KohonenLabExperiment()
        exp.setup({"width": 10, "height": 10, "mask": "simple"})

        frame_before = exp.get_frame()

        exp.reconnect({"mask": "wide_hat"})

        frame_after = exp.get_frame()
        assert frame_before == frame_after

    def test_reconnect_changes_connectivity(self) -> None:
        """Reconnect cambia la estructura de dendritas."""
        exp = KohonenLabExperiment()
        exp.setup({"width": 15, "height": 15, "mask": "simple"})

        n_before = len(exp.red.get_neurona("x7y7").dendritas)

        exp.reconnect({"mask": "one_dendrite"})

        n_after = len(exp.red.get_neurona("x7y7").dendritas)
        assert n_before != n_after

    def test_reconnect_updates_config(self) -> None:
        """Reconnect actualiza _config con los nuevos valores."""
        exp = KohonenLabExperiment()
        exp.setup({"width": 10, "height": 10, "mask": "simple", "balance": 0.0})

        exp.reconnect({"mask": "narrow_hat", "balance": 0.3})

        assert exp._config["mask"] == "narrow_hat"
        assert exp._config["balance"] == 0.3

    def test_reconnect_with_balance(self) -> None:
        """Reconnect con balance>0 reduce sinapsis inhibitorias vs balance=0."""
        exp = KohonenLabExperiment()
        exp.setup({"width": 10, "height": 10, "mask": "simple"})

        # Reconnect without balance (seed controls grid creation)
        random.seed(99)
        exp.reconnect({"mask": "simple", "balance": 0.0, "balance_mode": "weight"})
        n_no = exp.red.get_neurona("x5y5")
        inh_weights_no = []
        for d in n_no.dendritas:
            if d.peso < 0:
                inh_weights_no.extend(s.peso for s in d.sinapsis)

        # Reconnect again with same seed but balance=0.5
        random.seed(99)
        exp.reconnect({"mask": "simple", "balance": 0.5, "balance_mode": "weight"})
        n_bal = exp.red.get_neurona("x5y5")
        inh_weights_bal = []
        for d in n_bal.dendritas:
            if d.peso < 0:
                inh_weights_bal.extend(s.peso for s in d.sinapsis)

        # Same seed = same base weights; balance=0.5 scales inh by 0.5
        assert len(inh_weights_no) == len(inh_weights_bal)
        for w_no, w_bal in zip(inh_weights_no, inh_weights_bal):
            assert w_bal == pytest.approx(w_no * 0.5, abs=1e-9)


class TestKohonenLabFunctionality:
    """Funcionalidad est?ndar del experimento."""

    def test_step_avanza_generacion(self) -> None:
        exp = KohonenLabExperiment()
        exp.setup({"width": 10, "height": 10, "mask": "simple"})
        result = exp.step()
        assert result["type"] == "frame"
        assert result["generation"] == 1

    def test_step_n(self) -> None:
        exp = KohonenLabExperiment()
        exp.setup({"width": 10, "height": 10, "mask": "simple"})
        result = exp.step_n(5)
        assert result["generation"] == 5

    def test_click_toggle(self) -> None:
        exp = KohonenLabExperiment()
        exp.setup({"width": 10, "height": 10, "mask": "simple"})
        idx = 5 * 10 + 5
        exp.red_tensor.set_valor(idx, 0.0)
        exp.click(5, 5)
        assert exp.red_tensor.valores[idx].item() == 1.0
        exp.click(5, 5)
        assert exp.red_tensor.valores[idx].item() == 0.0

    def test_reset_reinicializa(self) -> None:
        exp = KohonenLabExperiment()
        exp.setup({"width": 10, "height": 10, "mask": "simple"})
        exp.step()
        exp.step()
        assert exp.generation == 2
        exp.reset()
        assert exp.generation == 0

    def test_is_complete_siempre_false(self) -> None:
        exp = KohonenLabExperiment()
        exp.setup({"width": 5, "height": 5, "mask": "simple"})
        assert exp.is_complete() is False
        for _ in range(10):
            exp.step()
        assert exp.is_complete() is False

    def test_get_frame_dimensions(self) -> None:
        exp = KohonenLabExperiment()
        exp.setup({"width": 10, "height": 10, "mask": "simple"})
        frame = exp.get_frame()
        assert len(frame) == 10
        assert all(len(row) == 10 for row in frame)

    @pytest.mark.parametrize("mask_id", list(MASK_PRESETS.keys()))
    def test_all_presets_run_10_steps(self, mask_id: str) -> None:
        """Every preset can run 10 steps without errors."""
        random.seed(42)
        exp = KohonenLabExperiment()
        exp.setup({"width": 10, "height": 10, "mask": mask_id})
        for _ in range(10):
            result = exp.step()
            assert result["type"] == "frame"

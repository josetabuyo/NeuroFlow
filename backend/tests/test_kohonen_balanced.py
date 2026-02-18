"""Tests para el Experimento Kohonen Balanceado.

Valida:
- Setup aplica el balanceo correctamente con la nueva semántica
  (target=0 no modifica nada, target>0 reduce inh, target<0 reduce exc)
- Misma funcionalidad que Kohonen (step, click, reset, etc.)
- Pesos sinápticos se mantienen en rango [0, 1]
"""

import random

import pytest
from core.neurona import Neurona
from experiments.kohonen_balanced import KohonenBalancedExperiment


def _get_syn_weights_by_type(neurona: Neurona, kind: str) -> list[float]:
    """Retorna pesos sinápticos de dendritas excitatorias o inhibitorias."""
    result = []
    for d in neurona.dendritas:
        if kind == "exc" and d.peso > 0:
            result.extend(s.peso for s in d.sinapsis)
        elif kind == "inh" and d.peso < 0:
            result.extend(s.peso for s in d.sinapsis)
    return result


class TestKohonenBalancedSetup:
    """Setup del experimento Kohonen Balanceado."""

    def test_setup_crea_red_30x30(self) -> None:
        """Setup crea red de 30x30 con 900 neuronas."""
        exp = KohonenBalancedExperiment()
        exp.setup({"width": 30, "height": 30})
        assert len(exp.red.neuronas) == 900

    def test_neurona_central_tiene_9_dendritas(self) -> None:
        """Neurona central tiene exactamente 9 dendritas (igual que Kohonen)."""
        exp = KohonenBalancedExperiment()
        exp.setup({"width": 30, "height": 30})
        neurona = exp.red.get_neurona("x15y15")
        assert len(neurona.dendritas) == 9

    def test_balance_cero_no_modifica_pesos(self) -> None:
        """Con balance=0.0, los pesos no se modifican (idéntico a Kohonen simple)."""
        from experiments.kohonen import KohonenExperiment

        random.seed(42)
        exp_balanced = KohonenBalancedExperiment()
        exp_balanced.setup({"width": 10, "height": 10, "balance": 0.0})

        random.seed(42)
        exp_simple = KohonenExperiment()
        exp_simple.setup({"width": 10, "height": 10})

        n_bal = exp_balanced.red.get_neurona("x5y5")
        n_sim = exp_simple.red.get_neurona("x5y5")

        # Los pesos sinápticos deben ser idénticos
        for db, ds in zip(n_bal.dendritas, n_sim.dendritas):
            for sb, ss in zip(db.sinapsis, ds.sinapsis):
                assert sb.peso == pytest.approx(ss.peso, abs=1e-9)

    def test_balance_positivo_reduce_inhibitorias(self) -> None:
        """Con balance>0, los pesos inhibitorios son menores que sin balance."""
        random.seed(42)
        exp_no_balance = KohonenBalancedExperiment()
        exp_no_balance.setup({"width": 10, "height": 10, "balance": 0.0})

        random.seed(42)
        exp_balanced = KohonenBalancedExperiment()
        exp_balanced.setup({"width": 10, "height": 10, "balance": 0.5})

        n_no = exp_no_balance.red.get_neurona("x5y5")
        n_bal = exp_balanced.red.get_neurona("x5y5")

        inh_no = _get_syn_weights_by_type(n_no, "inh")
        inh_bal = _get_syn_weights_by_type(n_bal, "inh")

        # Inhibitorias deben haberse reducido
        for w_no, w_bal in zip(inh_no, inh_bal):
            assert w_bal < w_no or w_no == 0.0

        # Excitatorias no cambian
        exc_no = _get_syn_weights_by_type(n_no, "exc")
        exc_bal = _get_syn_weights_by_type(n_bal, "exc")
        assert exc_no == exc_bal

    def test_balance_negativo_reduce_excitatorias(self) -> None:
        """Con balance<0, los pesos excitatorios son menores que sin balance."""
        random.seed(42)
        exp_no_balance = KohonenBalancedExperiment()
        exp_no_balance.setup({"width": 10, "height": 10, "balance": 0.0})

        random.seed(42)
        exp_balanced = KohonenBalancedExperiment()
        exp_balanced.setup({"width": 10, "height": 10, "balance": -0.5})

        n_no = exp_no_balance.red.get_neurona("x5y5")
        n_bal = exp_balanced.red.get_neurona("x5y5")

        exc_no = _get_syn_weights_by_type(n_no, "exc")
        exc_bal = _get_syn_weights_by_type(n_bal, "exc")

        # Excitatorias deben haberse reducido
        for w_no, w_bal in zip(exc_no, exc_bal):
            assert w_bal < w_no or w_no == 0.0

        # Inhibitorias no cambian
        inh_no = _get_syn_weights_by_type(n_no, "inh")
        inh_bal = _get_syn_weights_by_type(n_bal, "inh")
        assert inh_no == inh_bal

    def test_pesos_sinapticos_en_rango(self) -> None:
        """Todos los pesos sinápticos quedan en [0, 1] después del balanceo."""
        random.seed(42)
        exp = KohonenBalancedExperiment()
        exp.setup({"width": 10, "height": 10, "balance": 0.5})

        for neurona in exp.red.neuronas:
            for dendrita in neurona.dendritas:
                for sinapsis in dendrita.sinapsis:
                    assert 0.0 <= sinapsis.peso <= 1.0

    def test_balance_default_es_cero(self) -> None:
        """Si no se especifica balance en config, se usa 0.0 (sin cambio)."""
        from experiments.kohonen import KohonenExperiment

        random.seed(42)
        exp_default = KohonenBalancedExperiment()
        exp_default.setup({"width": 10, "height": 10})

        random.seed(42)
        exp_simple = KohonenExperiment()
        exp_simple.setup({"width": 10, "height": 10})

        n_def = exp_default.red.get_neurona("x5y5")
        n_sim = exp_simple.red.get_neurona("x5y5")

        for dd, ds in zip(n_def.dendritas, n_sim.dendritas):
            for sd, ss in zip(dd.sinapsis, ds.sinapsis):
                assert sd.peso == pytest.approx(ss.peso, abs=1e-9)


class TestKohonenBalancedFunctionality:
    """Funcionalidad del experimento (step, click, reset)."""

    def test_step_avanza_generacion(self) -> None:
        """Step incrementa la generación."""
        exp = KohonenBalancedExperiment()
        exp.setup({"width": 10, "height": 10, "balance": 0.0})

        result = exp.step()
        assert result["type"] == "frame"
        assert result["generation"] == 1

    def test_click_toggle(self) -> None:
        """Click alterna entre activar y desactivar."""
        exp = KohonenBalancedExperiment()
        exp.setup({"width": 10, "height": 10, "balance": 0.0})

        idx = 5 * 10 + 5  # y=5, x=5
        exp.red_tensor.set_valor(idx, 0.0)
        exp.click(5, 5)
        assert exp.red_tensor.valores[idx].item() == 1.0

        exp.click(5, 5)
        assert exp.red_tensor.valores[idx].item() == 0.0

    def test_reset_reinicializa(self) -> None:
        """Reset vuelve a generación 0 y reinicializa valores."""
        exp = KohonenBalancedExperiment()
        exp.setup({"width": 10, "height": 10, "balance": 0.0})

        exp.step()
        exp.step()
        assert exp.generation == 2

        exp.reset()
        assert exp.generation == 0

    def test_is_complete_siempre_false(self) -> None:
        """Kohonen balanceado nunca termina."""
        exp = KohonenBalancedExperiment()
        exp.setup({"width": 5, "height": 5, "balance": 0.0})
        assert exp.is_complete() is False
        for _ in range(10):
            exp.step()
        assert exp.is_complete() is False

    def test_get_frame_dimensiones_correctas(self) -> None:
        """get_frame retorna grilla con dimensiones correctas."""
        exp = KohonenBalancedExperiment()
        exp.setup({"width": 10, "height": 10, "balance": 0.0})
        frame = exp.get_frame()
        assert len(frame) == 10
        assert all(len(row) == 10 for row in frame)

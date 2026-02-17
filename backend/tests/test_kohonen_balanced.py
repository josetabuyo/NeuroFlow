"""Tests para el Experimento Kohonen Balanceado.

Valida:
- Setup aplica el balanceo correctamente
- El balance configurable funciona (target 0, positivo, negativo)
- Misma funcionalidad que Kohonen (step, click, reset, etc.)
- Pesos sinápticos se mantienen en rango [0, 1]
"""

import random

import pytest
from core.constructor import Constructor
from core.neurona import Neurona, NeuronaEntrada
from experiments.kohonen_balanced import KohonenBalancedExperiment


def _effective_sum(neurona: Neurona) -> float:
    """Calcula la suma de pesos efectivos (s.peso * d.peso) de una neurona."""
    total = 0.0
    for d in neurona.dendritas:
        for s in d.sinapsis:
            total += s.peso * d.peso
    return total


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

    def test_balance_cero_produce_equilibrio(self) -> None:
        """Con balance=0.0, neuronas centrales tienen pesos efectivos ~0."""
        random.seed(42)
        exp = KohonenBalancedExperiment()
        exp.setup({"width": 10, "height": 10, "balance": 0.0})

        # Neurona central (lejos de bordes, tiene todas las dendritas)
        neurona = exp.red.get_neurona("x5y5")
        balance = _effective_sum(neurona)
        assert balance == pytest.approx(0.0, abs=0.05)

    def test_balance_positivo_sesgo_excitatorio(self) -> None:
        """Con balance=0.5, los pesos efectivos dan positivo."""
        random.seed(42)
        exp = KohonenBalancedExperiment()
        exp.setup({"width": 10, "height": 10, "balance": 0.5})

        neurona = exp.red.get_neurona("x5y5")
        balance = _effective_sum(neurona)
        assert balance == pytest.approx(0.5, abs=0.05)

    def test_balance_negativo_sesgo_inhibitorio(self) -> None:
        """Con balance=-0.5, los pesos efectivos dan negativo."""
        random.seed(42)
        exp = KohonenBalancedExperiment()
        exp.setup({"width": 10, "height": 10, "balance": -0.5})

        neurona = exp.red.get_neurona("x5y5")
        balance = _effective_sum(neurona)
        assert balance == pytest.approx(-0.5, abs=0.05)

    def test_pesos_sinapticos_en_rango(self) -> None:
        """Todos los pesos sinápticos quedan en [0, 1] después del balanceo."""
        random.seed(42)
        exp = KohonenBalancedExperiment()
        exp.setup({"width": 10, "height": 10, "balance": 0.0})

        for neurona in exp.red.neuronas:
            for dendrita in neurona.dendritas:
                for sinapsis in dendrita.sinapsis:
                    assert 0.0 <= sinapsis.peso <= 1.0

    def test_balance_default_es_cero(self) -> None:
        """Si no se especifica balance en config, se usa 0.0."""
        random.seed(42)
        exp = KohonenBalancedExperiment()
        exp.setup({"width": 10, "height": 10})

        neurona = exp.red.get_neurona("x5y5")
        balance = _effective_sum(neurona)
        assert balance == pytest.approx(0.0, abs=0.05)


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

        exp.red.get_neurona("x5y5").activar_external(0.0)
        exp.click(5, 5)
        assert exp.red.get_neurona("x5y5").valor == 1.0

        exp.click(5, 5)
        assert exp.red.get_neurona("x5y5").valor == 0.0

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

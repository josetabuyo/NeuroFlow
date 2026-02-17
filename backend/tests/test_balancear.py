"""Tests para Constructor.balancear_pesos — balanceo de pesos excitatorios/inhibitorios."""

import pytest
from core.constructor import Constructor
from core.sinapsis import Sinapsis
from core.dendrita import Dendrita
from core.neurona import Neurona, NeuronaEntrada


def _effective_sum(neurona: Neurona) -> float:
    """Calcula la suma de pesos efectivos (s.peso * d.peso) de una neurona."""
    total = 0.0
    for d in neurona.dendritas:
        for s in d.sinapsis:
            total += s.peso * d.peso
    return total


def _build_neuron_with_dendrites(
    exc_weights: list[float],
    inh_weights: list[float],
    peso_exc: float = 1.0,
    peso_inh: float = -1.0,
) -> Neurona:
    """Crea una neurona con una dendrita excitatoria y una inhibitoria."""
    dummy = NeuronaEntrada(id="dummy")

    exc_sinapsis = [Sinapsis(neurona_entrante=dummy, peso=w) for w in exc_weights]
    inh_sinapsis = [Sinapsis(neurona_entrante=dummy, peso=w) for w in inh_weights]

    dendritas = [
        Dendrita(sinapsis=exc_sinapsis, peso=peso_exc),
        Dendrita(sinapsis=inh_sinapsis, peso=peso_inh),
    ]

    return Neurona(id="test", dendritas=dendritas)


class TestBalancearPesos:
    """Constructor.balancear_pesos: normaliza balance excitación/inhibición."""

    def test_target_cero_produce_balance_exacto(self) -> None:
        """Con target=0, la suma de pesos efectivos debe ser ~0."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.8, 0.9, 0.7],
            inh_weights=[0.3, 0.2],
        )
        # Before: unbalanced
        before = _effective_sum(neurona)
        assert before != pytest.approx(0.0, abs=0.01)

        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=0.0)

        after = _effective_sum(neurona)
        assert after == pytest.approx(0.0, abs=1e-9)

    def test_target_positivo_produce_sesgo_excitatorio(self) -> None:
        """Con target=0.1, la suma de pesos efectivos debe ser ~0.1."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.5, 0.6, 0.7],
            inh_weights=[0.4, 0.5, 0.6],
        )
        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=0.1)

        after = _effective_sum(neurona)
        assert after == pytest.approx(0.1, abs=1e-9)

    def test_target_negativo_produce_sesgo_inhibitorio(self) -> None:
        """Con target=-0.1, la suma de pesos efectivos debe ser ~-0.1."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.5, 0.6, 0.7],
            inh_weights=[0.4, 0.5, 0.6],
        )
        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=-0.1)

        after = _effective_sum(neurona)
        assert after == pytest.approx(-0.1, abs=1e-9)

    def test_neurona_sin_dendritas_no_falla(self) -> None:
        """Neuronas sin dendritas se ignoran sin error."""
        neurona = Neurona(id="vacia")
        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=0.0)
        assert neurona.dendritas == []

    def test_neurona_entrada_se_ignora(self) -> None:
        """NeuronaEntrada se salta sin modificar."""
        entrada = NeuronaEntrada(id="entrada")
        constructor = Constructor()
        constructor.balancear_pesos([entrada], target=0.0)
        assert entrada.dendritas == []

    def test_pesos_se_mantienen_en_rango(self) -> None:
        """Los pesos sinápticos quedan clampeados a [0, 1]."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.99, 0.95, 0.98],
            inh_weights=[0.1],
        )
        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=0.0)

        for d in neurona.dendritas:
            for s in d.sinapsis:
                assert 0.0 <= s.peso <= 1.0

    def test_balancea_multiples_neuronas(self) -> None:
        """Balancea correctamente una lista de varias neuronas."""
        n1 = _build_neuron_with_dendrites(
            exc_weights=[0.8, 0.7],
            inh_weights=[0.3, 0.2, 0.1],
        )
        n2 = _build_neuron_with_dendrites(
            exc_weights=[0.8, 0.7],
            inh_weights=[0.3, 0.2],
        )
        constructor = Constructor()
        constructor.balancear_pesos([n1, n2], target=0.0)

        assert _effective_sum(n1) == pytest.approx(0.0, abs=1e-9)
        assert _effective_sum(n2) == pytest.approx(0.0, abs=1e-9)

    def test_escala_siempre_hacia_abajo(self) -> None:
        """Los factores de escala son <= 1 (pesos nunca suben por encima del original)."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.6, 0.7, 0.8],
            inh_weights=[0.4, 0.5],
        )
        # Guardar pesos originales
        pesos_antes = [s.peso for d in neurona.dendritas for s in d.sinapsis]

        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=0.0)

        pesos_despues = [s.peso for d in neurona.dendritas for s in d.sinapsis]

        # Al menos uno debe haber cambiado (red no estaba balanceada)
        assert pesos_antes != pesos_despues
        # Ninguno subió por encima de su valor original
        for antes, despues in zip(pesos_antes, pesos_despues):
            assert despues <= antes + 1e-9

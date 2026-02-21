"""Tests for Constructor.balancear_pesos — Fuzzy OR-compatible scaling.

New semantics:
  target = 0.0  → no change (immediate return)
  target > 0    → scales inhibitory synapses by (1 - target)
  target < 0    → scales excitatory synapses by (1 + target)
  target = +1   → inhibition removed
  target = -1   → excitation removed
"""

import pytest
from core.constructor import Constructor
from core.sinapsis import Sinapsis
from core.dendrita import Dendrita
from core.neurona import Neurona, NeuronaEntrada


def _build_neuron_with_dendrites(
    exc_weights: list[float],
    inh_weights: list[float],
    peso_exc: float = 1.0,
    peso_inh: float = -1.0,
) -> Neurona:
    """Creates a neuron with one excitatory and one inhibitory dendrite."""
    dummy = NeuronaEntrada(id="dummy")

    exc_sinapsis = [Sinapsis(neurona_entrante=dummy, peso=w) for w in exc_weights]
    inh_sinapsis = [Sinapsis(neurona_entrante=dummy, peso=w) for w in inh_weights]

    dendritas = [
        Dendrita(sinapsis=exc_sinapsis, peso=peso_exc),
        Dendrita(sinapsis=inh_sinapsis, peso=peso_inh),
    ]

    return Neurona(id="test", dendritas=dendritas)


def _get_syn_weights(neurona: Neurona, kind: str) -> list[float]:
    """Returns synaptic weights from excitatory or inhibitory dendrites."""
    result = []
    for d in neurona.dendritas:
        if kind == "exc" and d.peso > 0:
            result.extend(s.peso for s in d.sinapsis)
        elif kind == "inh" and d.peso < 0:
            result.extend(s.peso for s in d.sinapsis)
    return result


class TestBalancearPesosNuevaSemantica:
    """Constructor.balancear_pesos: Fuzzy OR-compatible scaling."""

    def test_target_cero_no_modifica_nada(self) -> None:
        """With target=0, weights are not touched (immediate return)."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.8, 0.9, 0.7],
            inh_weights=[0.3, 0.2],
        )
        exc_before = _get_syn_weights(neurona, "exc")[:]
        inh_before = _get_syn_weights(neurona, "inh")[:]

        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=0.0)

        assert _get_syn_weights(neurona, "exc") == exc_before
        assert _get_syn_weights(neurona, "inh") == inh_before

    def test_target_positivo_reduce_inhibitorias(self) -> None:
        """With target>0, inhibitory synapses are scaled by (1 - target)."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.6, 0.8],
            inh_weights=[0.5, 0.4],
        )
        exc_before = _get_syn_weights(neurona, "exc")[:]
        inh_before = _get_syn_weights(neurona, "inh")[:]

        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=0.5)

        # Excitatory ones don't change
        assert _get_syn_weights(neurona, "exc") == exc_before
        # Inhibitory ones were scaled by 0.5
        inh_after = _get_syn_weights(neurona, "inh")
        for before, after in zip(inh_before, inh_after):
            assert after == pytest.approx(before * 0.5, abs=1e-9)

    def test_target_negativo_reduce_excitatorias(self) -> None:
        """With target<0, excitatory synapses are scaled by (1 + target)."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.6, 0.8],
            inh_weights=[0.5, 0.4],
        )
        exc_before = _get_syn_weights(neurona, "exc")[:]
        inh_before = _get_syn_weights(neurona, "inh")[:]

        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=-0.5)

        # Inhibitory ones don't change
        assert _get_syn_weights(neurona, "inh") == inh_before
        # Excitatory ones were scaled by 0.5
        exc_after = _get_syn_weights(neurona, "exc")
        for before, after in zip(exc_before, exc_after):
            assert after == pytest.approx(before * 0.5, abs=1e-9)

    def test_target_uno_elimina_inhibicion(self) -> None:
        """With target=+1, inhibition is scaled by 0.01 (~removed)."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.6],
            inh_weights=[0.5, 0.4],
        )
        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=1.0)

        inh_after = _get_syn_weights(neurona, "inh")
        for w in inh_after:
            assert w < 0.01  # nearly zero

    def test_target_menos_uno_elimina_excitacion(self) -> None:
        """With target=-1, excitation is scaled by 0.01 (~removed)."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.6, 0.8],
            inh_weights=[0.5],
        )
        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=-1.0)

        exc_after = _get_syn_weights(neurona, "exc")
        for w in exc_after:
            assert w < 0.01  # nearly zero

    def test_neurona_sin_dendritas_no_falla(self) -> None:
        """Neurons without dendrites are ignored without error."""
        neurona = Neurona(id="vacia")
        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=0.5)
        assert neurona.dendritas == []

    def test_neurona_entrada_se_ignora(self) -> None:
        """NeuronaEntrada is skipped without modification."""
        entrada = NeuronaEntrada(id="entrada")
        constructor = Constructor()
        constructor.balancear_pesos([entrada], target=0.5)
        assert entrada.dendritas == []

    def test_pesos_se_mantienen_en_rango(self) -> None:
        """Synaptic weights are clamped to [0, 1]."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.99, 0.95, 0.98],
            inh_weights=[0.1],
        )
        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=0.5)

        for d in neurona.dendritas:
            for s in d.sinapsis:
                assert 0.0 <= s.peso <= 1.0

    def test_balancea_multiples_neuronas(self) -> None:
        """Correctly balances a list of multiple neurons."""
        n1 = _build_neuron_with_dendrites(
            exc_weights=[0.8, 0.7],
            inh_weights=[0.3, 0.2, 0.1],
        )
        n2 = _build_neuron_with_dendrites(
            exc_weights=[0.8, 0.7],
            inh_weights=[0.3, 0.2],
        )
        inh_before_1 = _get_syn_weights(n1, "inh")[:]
        inh_before_2 = _get_syn_weights(n2, "inh")[:]

        constructor = Constructor()
        constructor.balancear_pesos([n1, n2], target=0.5)

        inh_after_1 = _get_syn_weights(n1, "inh")
        inh_after_2 = _get_syn_weights(n2, "inh")
        for before, after in zip(inh_before_1, inh_after_1):
            assert after == pytest.approx(before * 0.5, abs=1e-9)
        for before, after in zip(inh_before_2, inh_after_2):
            assert after == pytest.approx(before * 0.5, abs=1e-9)

    def test_escala_siempre_hacia_abajo(self) -> None:
        """Weights are always reduced or stay the same, never increase."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.6, 0.7, 0.8],
            inh_weights=[0.4, 0.5],
        )
        pesos_antes = [s.peso for d in neurona.dendritas for s in d.sinapsis]

        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=0.3)

        pesos_despues = [s.peso for d in neurona.dendritas for s in d.sinapsis]
        for antes, despues in zip(pesos_antes, pesos_despues):
            assert despues <= antes + 1e-9

    def test_factor_proporcional_a_target(self) -> None:
        """target=0.3 produces factor 0.7 on inhibitory ones."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.6],
            inh_weights=[0.5],
        )
        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=0.3)

        inh_after = _get_syn_weights(neurona, "inh")
        assert inh_after[0] == pytest.approx(0.5 * 0.7, abs=1e-9)


def _count_synapses(neurona: Neurona, kind: str) -> list[int]:
    """Returns list of synapse counts per dendrite of the given type."""
    result = []
    for d in neurona.dendritas:
        if kind == "exc" and d.peso > 0:
            result.append(len(d.sinapsis))
        elif kind == "inh" and d.peso < 0:
            result.append(len(d.sinapsis))
    return result


def _build_neuron_many_synapses(
    n_exc: int = 8,
    n_inh_dendritas: int = 8,
    n_inh_sinapsis: int = 9,
) -> Neurona:
    """Creates a neuron with 1 excitatory dendrite and N inhibitory ones."""
    dummy = NeuronaEntrada(id="dummy")

    exc_sinapsis = [Sinapsis(neurona_entrante=dummy, peso=0.6) for _ in range(n_exc)]
    dendritas = [Dendrita(sinapsis=exc_sinapsis, peso=1.0)]

    for _ in range(n_inh_dendritas):
        inh_sinapsis = [
            Sinapsis(neurona_entrante=dummy, peso=0.6)
            for _ in range(n_inh_sinapsis)
        ]
        dendritas.append(Dendrita(sinapsis=inh_sinapsis, peso=-1.0))

    return Neurona(id="test", dendritas=dendritas)


class TestBalancearSinapsis:
    """Constructor.balancear_sinapsis: synapse elimination."""

    def test_target_cero_no_elimina_sinapsis(self) -> None:
        """With target=0, no synapses are eliminated."""
        neurona = _build_neuron_many_synapses()
        counts_before = _count_synapses(neurona, "inh")

        constructor = Constructor()
        constructor.balancear_sinapsis([neurona], target=0.0)

        counts_after = _count_synapses(neurona, "inh")
        assert counts_before == counts_after

    def test_target_positivo_elimina_inhibitorias(self) -> None:
        """With target=0.5, each inhibitory dendrite loses ~50% of synapses."""
        neurona = _build_neuron_many_synapses(n_inh_sinapsis=10)

        constructor = Constructor()
        constructor.balancear_sinapsis([neurona], target=0.5)

        for d in neurona.dendritas:
            if d.peso < 0:
                assert len(d.sinapsis) == 5  # 10 - floor(10 * 0.5) = 5

    def test_target_positivo_no_toca_excitatorias(self) -> None:
        """With target>0, excitatory dendrites do not lose synapses."""
        neurona = _build_neuron_many_synapses(n_exc=8)
        exc_before = _count_synapses(neurona, "exc")

        constructor = Constructor()
        constructor.balancear_sinapsis([neurona], target=0.5)

        exc_after = _count_synapses(neurona, "exc")
        assert exc_before == exc_after

    def test_target_negativo_elimina_excitatorias(self) -> None:
        """With target=-0.5, excitatory dendrites lose ~50% of synapses."""
        neurona = _build_neuron_many_synapses(n_exc=10)

        constructor = Constructor()
        constructor.balancear_sinapsis([neurona], target=-0.5)

        exc_counts = _count_synapses(neurona, "exc")
        assert exc_counts[0] == 5  # 10 - floor(10 * 0.5) = 5

    def test_target_negativo_no_toca_inhibitorias(self) -> None:
        """With target<0, inhibitory dendrites do not lose synapses."""
        neurona = _build_neuron_many_synapses(n_inh_sinapsis=9)
        inh_before = _count_synapses(neurona, "inh")

        constructor = Constructor()
        constructor.balancear_sinapsis([neurona], target=-0.5)

        inh_after = _count_synapses(neurona, "inh")
        assert inh_before == inh_after

    def test_target_uno_deja_una_sinapsis(self) -> None:
        """With target=1.0, each inhibitory dendrite is left with 1 synapse."""
        neurona = _build_neuron_many_synapses(n_inh_sinapsis=9)

        constructor = Constructor()
        constructor.balancear_sinapsis([neurona], target=1.0)

        for d in neurona.dendritas:
            if d.peso < 0:
                assert len(d.sinapsis) == 1

    def test_siempre_al_menos_una_sinapsis(self) -> None:
        """All synapses are never eliminated from a dendrite."""
        neurona = _build_neuron_many_synapses(n_inh_sinapsis=2)

        constructor = Constructor()
        constructor.balancear_sinapsis([neurona], target=0.99)

        for d in neurona.dendritas:
            assert len(d.sinapsis) >= 1

    def test_neurona_entrada_se_ignora(self) -> None:
        """NeuronaEntrada is skipped without modification."""
        entrada = NeuronaEntrada(id="entrada")
        constructor = Constructor()
        constructor.balancear_sinapsis([entrada], target=0.5)
        assert entrada.dendritas == []

    def test_neurona_sin_dendritas_no_falla(self) -> None:
        """Neurons without dendrites are ignored without error."""
        neurona = Neurona(id="vacia")
        constructor = Constructor()
        constructor.balancear_sinapsis([neurona], target=0.5)
        assert neurona.dendritas == []

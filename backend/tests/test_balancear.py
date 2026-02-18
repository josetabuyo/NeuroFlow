"""Tests para Constructor.balancear_pesos — escalado Fuzzy OR-compatible.

Nueva semántica:
  target = 0.0  → sin cambio (retorno inmediato)
  target > 0    → escala sinapsis inhibitorias por (1 - target)
  target < 0    → escala sinapsis excitatorias por (1 + target)
  target = +1   → inhibición eliminada
  target = -1   → excitación eliminada
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
    """Crea una neurona con una dendrita excitatoria y una inhibitoria."""
    dummy = NeuronaEntrada(id="dummy")

    exc_sinapsis = [Sinapsis(neurona_entrante=dummy, peso=w) for w in exc_weights]
    inh_sinapsis = [Sinapsis(neurona_entrante=dummy, peso=w) for w in inh_weights]

    dendritas = [
        Dendrita(sinapsis=exc_sinapsis, peso=peso_exc),
        Dendrita(sinapsis=inh_sinapsis, peso=peso_inh),
    ]

    return Neurona(id="test", dendritas=dendritas)


def _get_syn_weights(neurona: Neurona, kind: str) -> list[float]:
    """Retorna pesos sinápticos de dendritas excitatorias o inhibitorias."""
    result = []
    for d in neurona.dendritas:
        if kind == "exc" and d.peso > 0:
            result.extend(s.peso for s in d.sinapsis)
        elif kind == "inh" and d.peso < 0:
            result.extend(s.peso for s in d.sinapsis)
    return result


class TestBalancearPesosNuevaSemantica:
    """Constructor.balancear_pesos: escalado Fuzzy OR-compatible."""

    def test_target_cero_no_modifica_nada(self) -> None:
        """Con target=0, los pesos no se tocan (retorno inmediato)."""
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
        """Con target>0, las sinapsis inhibitorias se escalan por (1 - target)."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.6, 0.8],
            inh_weights=[0.5, 0.4],
        )
        exc_before = _get_syn_weights(neurona, "exc")[:]
        inh_before = _get_syn_weights(neurona, "inh")[:]

        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=0.5)

        # Excitatorias no cambian
        assert _get_syn_weights(neurona, "exc") == exc_before
        # Inhibitorias se escalaron por 0.5
        inh_after = _get_syn_weights(neurona, "inh")
        for before, after in zip(inh_before, inh_after):
            assert after == pytest.approx(before * 0.5, abs=1e-9)

    def test_target_negativo_reduce_excitatorias(self) -> None:
        """Con target<0, las sinapsis excitatorias se escalan por (1 + target)."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.6, 0.8],
            inh_weights=[0.5, 0.4],
        )
        exc_before = _get_syn_weights(neurona, "exc")[:]
        inh_before = _get_syn_weights(neurona, "inh")[:]

        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=-0.5)

        # Inhibitorias no cambian
        assert _get_syn_weights(neurona, "inh") == inh_before
        # Excitatorias se escalaron por 0.5
        exc_after = _get_syn_weights(neurona, "exc")
        for before, after in zip(exc_before, exc_after):
            assert after == pytest.approx(before * 0.5, abs=1e-9)

    def test_target_uno_elimina_inhibicion(self) -> None:
        """Con target=+1, inhibición se escala por 0.01 (~eliminada)."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.6],
            inh_weights=[0.5, 0.4],
        )
        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=1.0)

        inh_after = _get_syn_weights(neurona, "inh")
        for w in inh_after:
            assert w < 0.01  # casi cero

    def test_target_menos_uno_elimina_excitacion(self) -> None:
        """Con target=-1, excitación se escala por 0.01 (~eliminada)."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.6, 0.8],
            inh_weights=[0.5],
        )
        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=-1.0)

        exc_after = _get_syn_weights(neurona, "exc")
        for w in exc_after:
            assert w < 0.01  # casi cero

    def test_neurona_sin_dendritas_no_falla(self) -> None:
        """Neuronas sin dendritas se ignoran sin error."""
        neurona = Neurona(id="vacia")
        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=0.5)
        assert neurona.dendritas == []

    def test_neurona_entrada_se_ignora(self) -> None:
        """NeuronaEntrada se salta sin modificar."""
        entrada = NeuronaEntrada(id="entrada")
        constructor = Constructor()
        constructor.balancear_pesos([entrada], target=0.5)
        assert entrada.dendritas == []

    def test_pesos_se_mantienen_en_rango(self) -> None:
        """Los pesos sinápticos quedan clampeados a [0, 1]."""
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
        """Balancea correctamente una lista de varias neuronas."""
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
        """Los pesos siempre se reducen o quedan igual, nunca suben."""
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
        """target=0.3 produce factor 0.7 en inhibitorias."""
        neurona = _build_neuron_with_dendrites(
            exc_weights=[0.6],
            inh_weights=[0.5],
        )
        constructor = Constructor()
        constructor.balancear_pesos([neurona], target=0.3)

        inh_after = _get_syn_weights(neurona, "inh")
        assert inh_after[0] == pytest.approx(0.5 * 0.7, abs=1e-9)

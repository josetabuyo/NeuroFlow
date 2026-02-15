"""Tests para Neurona y NeuronaEntrada."""

import pytest
from core.sinapsis import Sinapsis
from core.dendrita import Dendrita
from core.neurona import Neurona, NeuronaEntrada


class TestNeurona:
    """Neurona: fuzzy OR de dendritas, umbral de activación."""

    def test_neurona_con_dendrita_activa_se_activa(self) -> None:
        """Neurona con dendrita activa → se activa (valor=1)."""
        fuente = NeuronaEntrada(id="src")
        fuente.activar_external(1.0)

        sinapsis = [Sinapsis(neurona_entrante=fuente, peso=1.0)]
        dendrita = Dendrita(sinapsis=sinapsis, peso=1.0)

        neurona = Neurona(id="n1", dendritas=[dendrita], umbral=0.0)
        neurona.procesar()
        neurona.activar()
        assert neurona.valor == 1

    def test_neurona_sin_dendritas_activas_no_se_activa(self) -> None:
        """Neurona sin dendritas activas → no se activa (valor=0)."""
        fuente = NeuronaEntrada(id="src")
        fuente.activar_external(0.0)

        sinapsis = [Sinapsis(neurona_entrante=fuente, peso=1.0)]
        dendrita = Dendrita(sinapsis=sinapsis, peso=1.0)

        neurona = Neurona(id="n1", dendritas=[dendrita], umbral=0.0)
        neurona.procesar()
        neurona.activar()
        assert neurona.valor == 0

    def test_neurona_fuzzy_or_max_plus_min(self) -> None:
        """Neurona computa tension = max(dendritas) + min(dendritas)."""
        src1 = NeuronaEntrada(id="s1")
        src1.activar_external(1.0)
        src2 = NeuronaEntrada(id="s2")
        src2.activar_external(1.0)

        # Dendrita positiva (valor = 1.0)
        d_pos = Dendrita(
            sinapsis=[Sinapsis(neurona_entrante=src1, peso=1.0)],
            peso=1.0,
        )
        # Dendrita negativa (valor = -1.0)
        d_neg = Dendrita(
            sinapsis=[Sinapsis(neurona_entrante=src2, peso=1.0)],
            peso=-1.0,
        )

        neurona = Neurona(id="n1", dendritas=[d_pos, d_neg], umbral=0.0)
        neurona.procesar()
        # tension = max(1.0, -1.0) + min(1.0, -1.0) = 1.0 + (-1.0) = 0.0
        assert neurona.tension_superficial == pytest.approx(0.0)


class TestNeuronaEntrada:
    """NeuronaEntrada: sin dendritas, valor seteado externamente."""

    def test_acepta_valor_externo(self) -> None:
        """NeuronaEntrada acepta valor externo."""
        ne = NeuronaEntrada(id="entrada_1")
        ne.activar_external(1.0)
        assert ne.valor == 1.0

    def test_procesar_es_noop(self) -> None:
        """NeuronaEntrada.procesar() es no-op (no modifica valor)."""
        ne = NeuronaEntrada(id="entrada_1")
        ne.activar_external(0.7)
        ne.procesar()
        assert ne.valor == pytest.approx(0.7)

    def test_activar_es_noop(self) -> None:
        """NeuronaEntrada.activar() es no-op."""
        ne = NeuronaEntrada(id="entrada_1")
        ne.activar_external(0.5)
        ne.activar()
        # valor no cambia
        assert ne.valor == pytest.approx(0.5)

"""Tests para Dendrita — rama de entrada con múltiples sinapsis."""

import pytest
from core.sinapsis import Sinapsis
from core.dendrita import Dendrita
from core.neurona import NeuronaEntrada


class TestDendrita:
    """Dendrita.procesar() = avg(sinapsis.procesar()) * peso_dendrita."""

    def test_3_sinapsis_matching_valor_alto(self) -> None:
        """Dendrita con 3 sinapsis matching → valor alto (~1.0)."""
        # Tres fuentes activas (valor=1), sinapsis con peso=1 → cada una da 1.0
        fuentes = [NeuronaEntrada(id=f"src_{i}") for i in range(3)]
        for f in fuentes:
            f.activar_external(1.0)

        sinapsis = [Sinapsis(neurona_entrante=f, peso=1.0) for f in fuentes]
        d = Dendrita(sinapsis=sinapsis, peso=1.0)
        assert d.procesar() == pytest.approx(1.0)

    def test_3_sinapsis_no_matching_valor_bajo(self) -> None:
        """Dendrita con 3 sinapsis no-matching → valor bajo (~0.0)."""
        # Tres fuentes activas (valor=1), sinapsis con peso=0 → cada una da 0.0
        fuentes = [NeuronaEntrada(id=f"src_{i}") for i in range(3)]
        for f in fuentes:
            f.activar_external(1.0)

        sinapsis = [Sinapsis(neurona_entrante=f, peso=0.0) for f in fuentes]
        d = Dendrita(sinapsis=sinapsis, peso=1.0)
        assert d.procesar() == pytest.approx(0.0)

    def test_peso_negativo_invierte_resultado(self) -> None:
        """Dendrita con peso negativo invierte el resultado."""
        fuentes = [NeuronaEntrada(id=f"src_{i}") for i in range(3)]
        for f in fuentes:
            f.activar_external(1.0)

        sinapsis = [Sinapsis(neurona_entrante=f, peso=1.0) for f in fuentes]
        d = Dendrita(sinapsis=sinapsis, peso=-1.0)
        # avg = 1.0, * (-1.0) = -1.0
        assert d.procesar() == pytest.approx(-1.0)

    def test_1_sola_sinapsis_funciona(self) -> None:
        """Dendrita con 1 sola sinapsis funciona correctamente."""
        fuente = NeuronaEntrada(id="src")
        fuente.activar_external(1.0)

        sinapsis = [Sinapsis(neurona_entrante=fuente, peso=1.0)]
        d = Dendrita(sinapsis=sinapsis, peso=1.0)
        assert d.procesar() == pytest.approx(1.0)

    def test_peso_dendrita_rango(self) -> None:
        """El peso de la dendrita debe estar en [-1, 1]."""
        with pytest.raises(ValueError):
            Dendrita(sinapsis=[], peso=1.5)
        with pytest.raises(ValueError):
            Dendrita(sinapsis=[], peso=-1.5)

"""Tests para Sinapsis — conexión sináptica pesada."""

import pytest
from core.sinapsis import Sinapsis
from core.neurona import Neurona, NeuronaEntrada


class TestSinapsis:
    """Sinapsis.procesar() = 1 - |peso - neurona_entrante.valor|"""

    def test_peso_1_reconoce_entrada_1(self) -> None:
        """Sinapsis con peso=1 reconoce entrada=1 → valor = 1.0."""
        fuente = NeuronaEntrada(id="src")
        fuente.activar_external(1.0)
        s = Sinapsis(neurona_entrante=fuente, peso=1.0)
        assert s.procesar() == pytest.approx(1.0)

    def test_peso_0_reconoce_entrada_0(self) -> None:
        """Sinapsis con peso=0 reconoce entrada=0 → valor = 1.0."""
        fuente = NeuronaEntrada(id="src")
        fuente.activar_external(0.0)
        s = Sinapsis(neurona_entrante=fuente, peso=0.0)
        assert s.procesar() == pytest.approx(1.0)

    def test_peso_1_rechaza_entrada_0(self) -> None:
        """Sinapsis con peso=1 rechaza entrada=0 → valor = 0.0."""
        fuente = NeuronaEntrada(id="src")
        fuente.activar_external(0.0)
        s = Sinapsis(neurona_entrante=fuente, peso=1.0)
        assert s.procesar() == pytest.approx(0.0)

    def test_peso_0_rechaza_entrada_1(self) -> None:
        """Sinapsis con peso=0 rechaza entrada=1 → valor = 0.0."""
        fuente = NeuronaEntrada(id="src")
        fuente.activar_external(1.0)
        s = Sinapsis(neurona_entrante=fuente, peso=0.0)
        assert s.procesar() == pytest.approx(0.0)

    def test_peso_siempre_en_rango(self) -> None:
        """El peso de la sinapsis siempre debe estar en [0, 1]."""
        fuente = NeuronaEntrada(id="src")
        with pytest.raises(ValueError):
            Sinapsis(neurona_entrante=fuente, peso=-0.1)
        with pytest.raises(ValueError):
            Sinapsis(neurona_entrante=fuente, peso=1.1)

    def test_valor_intermedio(self) -> None:
        """Sinapsis con peso=0.7 y entrada=0.5 → 1 - |0.7 - 0.5| = 0.8."""
        fuente = NeuronaEntrada(id="src")
        fuente.activar_external(0.5)
        s = Sinapsis(neurona_entrante=fuente, peso=0.7)
        assert s.procesar() == pytest.approx(0.8)

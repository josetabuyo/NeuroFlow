"""Tests para Region — agrupación de neuronas por referencia."""

import pytest
from core.neurona import Neurona, NeuronaEntrada
from core.red import Red
from core.region import Region


class TestRegion:
    """Region: grupo nombrado de referencias a neuronas."""

    def test_region_agrupa_neuronas_por_referencia(self) -> None:
        """Region agrupa neuronas por referencia."""
        n1 = Neurona(id="n1")
        n2 = Neurona(id="n2")
        region = Region(nombre="test")
        region.agregar(n1)
        region.agregar(n2)
        assert len(region.neuronas) == 2
        assert region.neuronas["n1"] is n1
        assert region.neuronas["n2"] is n2

    def test_region_puede_agregar_y_consultar(self) -> None:
        """Region puede agregar/consultar neuronas."""
        ne = NeuronaEntrada(id="e1")
        ne.activar_external(1.0)
        region = Region(nombre="entrada")
        region.agregar(ne)
        assert "e1" in region.ids()
        assert region.valores() == [1.0]

    def test_region_no_afecta_valores_de_red(self) -> None:
        """Region no afecta los valores de las neuronas en la Red."""
        n = NeuronaEntrada(id="n1")
        n.activar_external(1.0)
        red = Red(neuronas=[n])
        region = Region(nombre="grupo")
        region.agregar(n)

        assert n.valor == 1.0
        assert red.get_neurona("n1").valor == 1.0

    def test_multiples_regiones_misma_red(self) -> None:
        """Múltiples regiones pueden compartir la misma Red."""
        n1 = NeuronaEntrada(id="n1")
        n2 = NeuronaEntrada(id="n2")
        n1.activar_external(1.0)
        n2.activar_external(0.0)

        red = Red(neuronas=[n1, n2])

        region_a = Region(nombre="A")
        region_a.agregar(n1)

        region_b = Region(nombre="B")
        region_b.agregar(n2)

        # Ambas regiones apuntan a neuronas de la misma red
        assert region_a.neuronas["n1"] is red.get_neurona("n1")
        assert region_b.neuronas["n2"] is red.get_neurona("n2")

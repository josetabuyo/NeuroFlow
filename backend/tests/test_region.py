"""Tests for Region — neuron grouping by reference."""

import pytest
from core.neurona import Neurona, NeuronaEntrada
from core.brain import Brain
from core.region import Region


class TestRegion:
    """Region: named group of neuron references."""

    def test_region_agrupa_neuronas_por_referencia(self) -> None:
        """Region groups neurons by reference."""
        n1 = Neurona(id="n1")
        n2 = Neurona(id="n2")
        region = Region(nombre="test")
        region.agregar(n1)
        region.agregar(n2)
        assert len(region.neuronas) == 2
        assert region.neuronas["n1"] is n1
        assert region.neuronas["n2"] is n2

    def test_region_puede_agregar_y_consultar(self) -> None:
        """Region can add/query neurons."""
        ne = NeuronaEntrada(id="e1")
        ne.activar_external(1.0)
        region = Region(nombre="entrada")
        region.agregar(ne)
        assert "e1" in region.ids()
        assert region.valores() == [1.0]

    def test_region_no_afecta_valores_de_brain(self) -> None:
        """Region does not affect neuron values in the Brain."""
        n = NeuronaEntrada(id="n1")
        n.activar_external(1.0)
        brain = Brain(neuronas=[n])
        region = Region(nombre="grupo")
        region.agregar(n)

        assert n.valor == 1.0
        assert brain.get_neurona("n1").valor == 1.0

    def test_multiples_regiones_misma_brain(self) -> None:
        """Multiple regions can share the same Brain."""
        n1 = NeuronaEntrada(id="n1")
        n2 = NeuronaEntrada(id="n2")
        n1.activar_external(1.0)
        n2.activar_external(0.0)

        brain = Brain(neuronas=[n1, n2])

        region_a = Region(nombre="A")
        region_a.agregar(n1)

        region_b = Region(nombre="B")
        region_b.agregar(n2)

        # Both regions point to neurons in the same network
        assert region_a.neuronas["n1"] is brain.get_neurona("n1")
        assert region_b.neuronas["n2"] is brain.get_neurona("n2")

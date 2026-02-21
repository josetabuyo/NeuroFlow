"""Tests for Constructor — factory for building networks + regions."""

import pytest
from core.constructor import Constructor
from core.neurona import Neurona, NeuronaEntrada
from core.red import Red
from core.region import Region


class TestConstructor:
    """Constructor: creates neurons, regions, connectivity."""

    def test_crear_grilla_de_neuronas(self) -> None:
        """Constructor correctly creates neuron grid."""
        constructor = Constructor()
        red, regiones = constructor.crear_grilla(
            width=5,
            height=5,
            filas_entrada=[4],
            filas_salida=[0],
        )
        assert len(red.neuronas) == 25  # 5x5

    def test_crear_regiones_separadas_de_red(self) -> None:
        """Constructor creates regions separate from the Red."""
        constructor = Constructor()
        red, regiones = constructor.crear_grilla(
            width=5,
            height=5,
            filas_entrada=[4],
            filas_salida=[0],
        )
        assert "entrada" in regiones
        assert "salida" in regiones
        assert "interna" in regiones
        # Red does not know about regions
        assert not hasattr(red, "regiones")

    def test_neuronas_entrada_son_neurona_entrada(self) -> None:
        """Neurons in the input row are NeuronaEntrada."""
        constructor = Constructor()
        red, regiones = constructor.crear_grilla(
            width=5,
            height=5,
            filas_entrada=[4],
            filas_salida=[0],
        )
        for neurona in regiones["entrada"].neuronas.values():
            assert isinstance(neurona, NeuronaEntrada)

    def test_conectar_neuronas_con_dendritas_y_sinapsis(self) -> None:
        """Constructor connects neurons with dendrites and synapses."""
        constructor = Constructor()
        red, regiones = constructor.crear_grilla(
            width=5,
            height=3,
            filas_entrada=[2],
            filas_salida=[0],
        )
        # Connect row 1 to row 2 (input) with 3-neighborhood
        mascara = [(-1, 1), (0, 1), (1, 1)]  # left, center, right of lower row
        constructor.conectar_filas(
            red=red,
            fila_destino=1,
            width=5,
            height=3,
            mascara_relativa=mascara,
            regla_pesos=[[1.0, 1.0, 1.0]],  # One dendrite that recognizes all 1s
        )

        # The central neuron of row 1 must have at least one dendrite
        neurona_centro = red.get_neurona("x2y1")
        assert len(neurona_centro.dendritas) > 0
        # That dendrite must have 3 synapses
        assert len(neurona_centro.dendritas[0].sinapsis) == 3

    def test_aplicar_mascara_de_conexion_relativa(self) -> None:
        """Constructor applies relative connection mask."""
        constructor = Constructor()
        red, regiones = constructor.crear_grilla(
            width=5,
            height=3,
            filas_entrada=[2],
            filas_salida=[0],
        )
        mascara = [(-1, 1), (0, 1), (1, 1)]
        constructor.conectar_filas(
            red=red,
            fila_destino=1,
            width=5,
            height=3,
            mascara_relativa=mascara,
            regla_pesos=[[1.0, 0.0, 1.0]],
        )

        neurona = red.get_neurona("x2y1")
        dendrita = neurona.dendritas[0]
        # Should be connected to x1y2, x2y2, x3y2
        ids_conectados = [s.neurona_entrante.id for s in dendrita.sinapsis]
        assert "x1y2" in ids_conectados
        assert "x2y2" in ids_conectados
        assert "x3y2" in ids_conectados

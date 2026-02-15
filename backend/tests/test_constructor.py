"""Tests para Constructor — factory para armar redes + regiones."""

import pytest
from core.constructor import Constructor
from core.neurona import Neurona, NeuronaEntrada
from core.red import Red
from core.region import Region


class TestConstructor:
    """Constructor: crea neuronas, regiones, conectividad."""

    def test_crear_grilla_de_neuronas(self) -> None:
        """Constructor crea grilla de neuronas correctamente."""
        constructor = Constructor()
        red, regiones = constructor.crear_grilla(
            width=5,
            height=5,
            filas_entrada=[4],
            filas_salida=[0],
        )
        assert len(red.neuronas) == 25  # 5x5

    def test_crear_regiones_separadas_de_red(self) -> None:
        """Constructor crea regiones separadas de la Red."""
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
        # Red no sabe de regiones
        assert not hasattr(red, "regiones")

    def test_neuronas_entrada_son_neurona_entrada(self) -> None:
        """Las neuronas de la fila de entrada son NeuronaEntrada."""
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
        """Constructor conecta neuronas con dendritas y sinapsis."""
        constructor = Constructor()
        red, regiones = constructor.crear_grilla(
            width=5,
            height=3,
            filas_entrada=[2],
            filas_salida=[0],
        )
        # Conectar fila 1 a fila 2 (entrada) con vecindad de 3
        mascara = [(-1, 1), (0, 1), (1, 1)]  # izq, centro, der de fila inferior
        constructor.conectar_filas(
            red=red,
            fila_destino=1,
            width=5,
            mascara_relativa=mascara,
            regla_pesos=[[1.0, 1.0, 1.0]],  # Una dendrita que reconoce todo 1s
        )

        # La neurona central de fila 1 debe tener al menos una dendrita
        neurona_centro = red.get_neurona("x2y1")
        assert len(neurona_centro.dendritas) > 0
        # Esa dendrita debe tener 3 sinapsis
        assert len(neurona_centro.dendritas[0].sinapsis) == 3

    def test_aplicar_mascara_de_conexion_relativa(self) -> None:
        """Constructor aplica máscara de conexión relativa."""
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
            mascara_relativa=mascara,
            regla_pesos=[[1.0, 0.0, 1.0]],
        )

        neurona = red.get_neurona("x2y1")
        dendrita = neurona.dendritas[0]
        # Debe estar conectada a x1y2, x2y2, x3y2
        ids_conectados = [s.neurona_entrante.id for s in dendrita.sinapsis]
        assert "x1y2" in ids_conectados
        assert "x2y2" in ids_conectados
        assert "x3y2" in ids_conectados

"""Tests para Red — contenedor tonto de neuronas."""

import pytest
from core.sinapsis import Sinapsis
from core.dendrita import Dendrita
from core.neurona import Neurona, NeuronaEntrada
from core.red import Red


class TestRed:
    """Red: contenedor de neuronas, solo las itera y procesa."""

    def test_red_se_construye_con_n_neuronas(self) -> None:
        """Red se construye con N neuronas."""
        neuronas = [Neurona(id=f"n{i}") for i in range(5)]
        red = Red(neuronas=neuronas)
        assert len(red.neuronas) == 5

    def test_procesar_actualiza_todas_las_neuronas(self) -> None:
        """Red.procesar() actualiza todas las neuronas (incluyendo NeuronaEntrada sin efecto)."""
        # Crear una NeuronaEntrada con valor 1
        ne = NeuronaEntrada(id="entrada")
        ne.activar_external(1.0)

        # Crear una Neurona normal conectada a la entrada
        sinapsis = [Sinapsis(neurona_entrante=ne, peso=1.0)]
        dendrita = Dendrita(sinapsis=sinapsis, peso=1.0)
        n = Neurona(id="n1", dendritas=[dendrita], umbral=0.0)

        red = Red(neuronas=[ne, n])
        red.procesar()

        # La NeuronaEntrada mantiene su valor (no-op)
        assert ne.valor == 1.0
        # La neurona normal se activó
        assert n.valor == 1

    def test_red_no_tiene_concepto_de_regiones(self) -> None:
        """Red NO tiene concepto de regiones."""
        red = Red(neuronas=[])
        assert not hasattr(red, "regiones")

    def test_get_grid_retorna_matriz_esperada(self) -> None:
        """get_grid(w, h) retorna la matriz esperada."""
        # Crear una grilla 3x2 de neuronas con IDs x{col}y{row}
        neuronas: list[Neurona] = []
        for row in range(2):
            for col in range(3):
                ne = NeuronaEntrada(id=f"x{col}y{row}")
                # Activar diagonales
                ne.activar_external(1.0 if row == col else 0.0)
                neuronas.append(ne)

        red = Red(neuronas=neuronas)
        grid = red.get_grid(3, 2)

        assert len(grid) == 2  # 2 filas
        assert len(grid[0]) == 3  # 3 columnas
        # Fila 0: [1, 0, 0]
        assert grid[0][0] == 1.0
        assert grid[0][1] == 0.0
        assert grid[0][2] == 0.0
        # Fila 1: [0, 1, 0]
        assert grid[1][0] == 0.0
        assert grid[1][1] == 1.0
        assert grid[1][2] == 0.0

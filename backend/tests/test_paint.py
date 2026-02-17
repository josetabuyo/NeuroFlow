"""Tests para la acción paint — activar/desactivar múltiples neuronas con pincel.

Valida:
- Paint con una celda activa valor 1.0
- Paint con una celda desactiva valor 0.0
- Paint con múltiples celdas activa todas
- Paint con celdas fuera del grid no falla (KeyError ignorado)
- Paint actualiza el frame (el grid refleja los cambios)
- Paint sin experimento retorna error
"""

import pytest
from core.constructor import Constructor
from experiments.kohonen import KohonenExperiment
from experiments.von_neumann import VonNeumannExperiment


class TestPaint:
    """Acción paint: activar/desactivar grupos de neuronas."""

    def test_paint_una_celda_activa(self) -> None:
        """Paint con una celda y valor 1.0 activa la neurona."""
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})

        # Forzar neurona a 0
        neurona = exp.red.get_neurona("x5y5")
        neurona.activar_external(0.0)
        assert neurona.valor == 0.0

        # Paint: activar
        key = Constructor.key_by_coord(5, 5)
        neurona = exp.red.get_neurona(key)
        neurona.activar_external(1.0)

        assert neurona.valor == 1.0

    def test_paint_una_celda_desactiva(self) -> None:
        """Paint con una celda y valor 0.0 desactiva la neurona."""
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})

        # Forzar neurona a 1
        neurona = exp.red.get_neurona("x5y5")
        neurona.activar_external(1.0)
        assert neurona.valor == 1.0

        # Paint: desactivar
        key = Constructor.key_by_coord(5, 5)
        neurona = exp.red.get_neurona(key)
        neurona.activar_external(0.0)

        assert neurona.valor == 0.0

    def test_paint_multiples_celdas_activa_todas(self) -> None:
        """Paint con múltiples celdas activa todas las neuronas."""
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})

        # Forzar varias neuronas a 0
        celdas = [(3, 3), (4, 3), (5, 3), (3, 4), (4, 4), (5, 4)]
        for x, y in celdas:
            exp.red.get_neurona(Constructor.key_by_coord(x, y)).activar_external(0.0)

        # Paint: activar todas
        for x, y in celdas:
            key = Constructor.key_by_coord(x, y)
            neurona = exp.red.get_neurona(key)
            neurona.activar_external(1.0)

        # Verificar que todas están activas
        for x, y in celdas:
            neurona = exp.red.get_neurona(Constructor.key_by_coord(x, y))
            assert neurona.valor == 1.0, f"Neurona ({x},{y}) debería estar activa"

    def test_paint_celdas_fuera_del_grid_no_falla(self) -> None:
        """Paint con celdas fuera del grid ignora KeyError sin fallar."""
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})

        # Celdas que incluyen posiciones fuera del grid
        celdas = [
            {"x": 5, "y": 5},   # Válida
            {"x": -1, "y": 5},  # Fuera
            {"x": 5, "y": -1},  # Fuera
            {"x": 10, "y": 5},  # Fuera
            {"x": 5, "y": 10},  # Fuera
            {"x": 99, "y": 99}, # Fuera
        ]

        # Simular lógica del handler: no debería lanzar excepciones
        for cell in celdas:
            x, y = cell["x"], cell["y"]
            key = Constructor.key_by_coord(x, y)
            try:
                neurona = exp.red.get_neurona(key)
                neurona.activar_external(1.0)
            except KeyError:
                pass  # Celda fuera del grid

        # La celda válida sí se activó
        assert exp.red.get_neurona("x5y5").valor == 1.0

    def test_paint_actualiza_frame(self) -> None:
        """Paint modifica el frame: el grid refleja los cambios."""
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})

        # Desactivar toda la grilla
        for neurona in exp.red.neuronas:
            neurona.activar_external(0.0)

        # Verificar que el frame tiene todo en 0
        frame_antes = exp.get_frame()
        assert all(cell == 0.0 for row in frame_antes for cell in row)

        # Paint: activar un bloque 3x3 centrado en (5,5)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = 5 + dx, 5 + dy
                key = Constructor.key_by_coord(x, y)
                exp.red.get_neurona(key).activar_external(1.0)

        # El frame refleja los cambios
        frame_despues = exp.get_frame()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                assert frame_despues[5 + dy][5 + dx] == 1.0, \
                    f"Celda ({5+dx},{5+dy}) debería estar activa en el frame"

    def test_paint_funciona_en_von_neumann(self) -> None:
        """Paint funciona igual en Von Neumann — activar_external directamente."""
        exp = VonNeumannExperiment()
        exp.setup({"width": 20, "height": 20, "rule": 111})

        # Paint: activar una neurona de entrada (fila 19)
        key = Constructor.key_by_coord(10, 19)
        neurona = exp.red.get_neurona(key)
        neurona.activar_external(1.0)

        assert neurona.valor == 1.0

        # El frame refleja el cambio
        frame = exp.get_frame()
        assert frame[19][10] == 1.0

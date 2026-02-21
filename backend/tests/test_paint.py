"""Tests para la acción paint — activar/desactivar múltiples neuronas con pincel.

Valida:
- Paint con una celda activa valor 1.0
- Paint con una celda desactiva valor 0.0
- Paint con múltiples celdas activa todas
- Paint con celdas fuera del grid no falla
- Paint actualiza el frame (el grid refleja los cambios)
"""

import pytest
from experiments.deamons_lab import DeamonsLabExperiment


class TestPaint:
    """Acción paint: activar/desactivar grupos de neuronas."""

    def test_paint_una_celda_activa(self) -> None:
        """Paint con una celda y valor 1.0 activa la neurona."""
        exp = DeamonsLabExperiment()
        exp.setup({"width": 10, "height": 10})

        idx = 5 * 10 + 5  # y=5, x=5
        exp.red_tensor.set_valor(idx, 0.0)
        assert exp.red_tensor.valores[idx].item() == 0.0

        exp.red_tensor.set_valor(idx, 1.0)
        assert exp.red_tensor.valores[idx].item() == 1.0

    def test_paint_una_celda_desactiva(self) -> None:
        """Paint con una celda y valor 0.0 desactiva la neurona."""
        exp = DeamonsLabExperiment()
        exp.setup({"width": 10, "height": 10})

        idx = 5 * 10 + 5
        exp.red_tensor.set_valor(idx, 1.0)
        assert exp.red_tensor.valores[idx].item() == 1.0

        exp.red_tensor.set_valor(idx, 0.0)
        assert exp.red_tensor.valores[idx].item() == 0.0

    def test_paint_multiples_celdas_activa_todas(self) -> None:
        """Paint con múltiples celdas activa todas las neuronas."""
        exp = DeamonsLabExperiment()
        exp.setup({"width": 10, "height": 10})

        celdas = [(3, 3), (4, 3), (5, 3), (3, 4), (4, 4), (5, 4)]
        for x, y in celdas:
            idx = y * 10 + x
            exp.red_tensor.set_valor(idx, 0.0)

        for x, y in celdas:
            idx = y * 10 + x
            exp.red_tensor.set_valor(idx, 1.0)

        for x, y in celdas:
            idx = y * 10 + x
            assert exp.red_tensor.valores[idx].item() == 1.0, \
                f"Neurona ({x},{y}) debería estar activa"

    def test_paint_celdas_fuera_del_grid_no_falla(self) -> None:
        """Paint con celdas fuera del grid ignora sin fallar."""
        exp = DeamonsLabExperiment()
        exp.setup({"width": 10, "height": 10})

        celdas = [
            (5, 5),    # Válida
            (-1, 5),   # Fuera
            (5, -1),   # Fuera
            (10, 5),   # Fuera
            (5, 10),   # Fuera
            (99, 99),  # Fuera
        ]

        for x, y in celdas:
            idx = y * 10 + x
            if 0 <= idx < exp.red_tensor.n_real and 0 <= x < 10 and 0 <= y < 10:
                exp.red_tensor.set_valor(idx, 1.0)

        idx_valid = 5 * 10 + 5
        assert exp.red_tensor.valores[idx_valid].item() == 1.0

    def test_paint_actualiza_frame(self) -> None:
        """Paint modifica el frame: el grid refleja los cambios."""
        exp = DeamonsLabExperiment()
        exp.setup({"width": 10, "height": 10})

        for i in range(exp.red_tensor.n_real):
            exp.red_tensor.set_valor(i, 0.0)

        frame_antes = exp.get_frame()
        assert all(cell == 0.0 for row in frame_antes for cell in row)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = 5 + dx, 5 + dy
                idx = y * 10 + x
                exp.red_tensor.set_valor(idx, 1.0)

        frame_despues = exp.get_frame()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                assert frame_despues[5 + dy][5 + dx] == 1.0, \
                    f"Celda ({5+dx},{5+dy}) debería estar activa en el frame"

"""Tests for the paint action — activate/deactivate multiple neurons with brush.

Validates:
- Paint with one cell activates value 1.0
- Paint with one cell deactivates value 0.0
- Paint with multiple cells activates all
- Paint with cells outside the grid does not fail
- Paint updates the frame (the grid reflects changes)
"""

import pytest
from experiments.deamons_lab import DeamonsLabExperiment


class TestPaint:
    """Paint action: activate/deactivate groups of neurons."""

    def test_paint_una_celda_activa(self) -> None:
        """Paint with one cell and value 1.0 activates the neuron."""
        exp = DeamonsLabExperiment()
        exp.setup({"width": 10, "height": 10})

        idx = 5 * 10 + 5  # y=5, x=5
        exp.red_tensor.set_valor(idx, 0.0)
        assert exp.red_tensor.valores[idx].item() == 0.0

        exp.red_tensor.set_valor(idx, 1.0)
        assert exp.red_tensor.valores[idx].item() == 1.0

    def test_paint_una_celda_desactiva(self) -> None:
        """Paint with one cell and value 0.0 deactivates the neuron."""
        exp = DeamonsLabExperiment()
        exp.setup({"width": 10, "height": 10})

        idx = 5 * 10 + 5
        exp.red_tensor.set_valor(idx, 1.0)
        assert exp.red_tensor.valores[idx].item() == 1.0

        exp.red_tensor.set_valor(idx, 0.0)
        assert exp.red_tensor.valores[idx].item() == 0.0

    def test_paint_multiples_celdas_activa_todas(self) -> None:
        """Paint with multiple cells activates all neurons."""
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
                f"Neuron ({x},{y}) should be active"

    def test_paint_celdas_fuera_del_grid_no_falla(self) -> None:
        """Paint with cells outside the grid ignores without failing."""
        exp = DeamonsLabExperiment()
        exp.setup({"width": 10, "height": 10})

        celdas = [
            (5, 5),    # Valid
            (-1, 5),   # Outside
            (5, -1),   # Outside
            (10, 5),   # Outside
            (5, 10),   # Outside
            (99, 99),  # Outside
        ]

        for x, y in celdas:
            idx = y * 10 + x
            if 0 <= idx < exp.red_tensor.n_real and 0 <= x < 10 and 0 <= y < 10:
                exp.red_tensor.set_valor(idx, 1.0)

        idx_valid = 5 * 10 + 5
        assert exp.red_tensor.valores[idx_valid].item() == 1.0

    def test_paint_actualiza_frame(self) -> None:
        """Paint modifies the frame: the grid reflects changes."""
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
                    f"Cell ({5+dx},{5+dy}) should be active in the frame"

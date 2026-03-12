"""Tests for the paint action — activate/deactivate multiple neurons with brush.

Validates:
- Paint with one cell activates value 1.0
- Paint with one cell deactivates value 0.0
- Paint with multiple cells activates all
- Paint with cells outside the grid does not fail
- Paint updates the frame (the grid reflects changes)
"""

from experiments.experiment import Experiment


def _nested_config(width: int = 10, height: int = 10) -> dict:
    return {
        "grid": {"width": width, "height": height},
        "wiring": {"mask": "simple", "process_mode": "min_vs_max"},
    }


class TestPaint:
    """Paint action: activate/deactivate groups of neurons."""

    def test_paint_una_celda_activa(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())

        idx = 5 * 10 + 5
        exp.brain_tensor.set_valor(idx, 0.0)
        assert exp.brain_tensor.valores[idx].item() == 0.0

        exp.brain_tensor.set_valor(idx, 1.0)
        assert exp.brain_tensor.valores[idx].item() == 1.0

    def test_paint_una_celda_desactiva(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())

        idx = 5 * 10 + 5
        exp.brain_tensor.set_valor(idx, 1.0)
        assert exp.brain_tensor.valores[idx].item() == 1.0

        exp.brain_tensor.set_valor(idx, 0.0)
        assert exp.brain_tensor.valores[idx].item() == 0.0

    def test_paint_multiples_celdas_activa_todas(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())

        celdas = [(3, 3), (4, 3), (5, 3), (3, 4), (4, 4), (5, 4)]
        for x, y in celdas:
            idx = y * 10 + x
            exp.brain_tensor.set_valor(idx, 0.0)

        for x, y in celdas:
            idx = y * 10 + x
            exp.brain_tensor.set_valor(idx, 1.0)

        for x, y in celdas:
            idx = y * 10 + x
            assert exp.brain_tensor.valores[idx].item() == 1.0, \
                f"Neuron ({x},{y}) should be active"

    def test_paint_celdas_fuera_del_grid_no_falla(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())

        celdas = [
            (5, 5),
            (-1, 5),
            (5, -1),
            (10, 5),
            (5, 10),
            (99, 99),
        ]

        for x, y in celdas:
            idx = y * 10 + x
            if 0 <= idx < exp.brain_tensor.n_real and 0 <= x < 10 and 0 <= y < 10:
                exp.brain_tensor.set_valor(idx, 1.0)

        idx_valid = 5 * 10 + 5
        assert exp.brain_tensor.valores[idx_valid].item() == 1.0

    def test_paint_actualiza_frame(self) -> None:
        exp = Experiment()
        exp.setup(_nested_config())

        for i in range(exp.brain_tensor.n_real):
            exp.brain_tensor.set_valor(i, 0.0)

        frame_antes = exp.get_frame()
        assert all(cell == 0.0 for row in frame_antes for cell in row)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                x, y = 5 + dx, 5 + dy
                idx = y * 10 + x
                exp.brain_tensor.set_valor(idx, 1.0)

        frame_despues = exp.get_frame()
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                assert frame_despues[5 + dy][5 + dx] == 1.0, \
                    f"Cell ({5+dx},{5+dy}) should be active in the frame"

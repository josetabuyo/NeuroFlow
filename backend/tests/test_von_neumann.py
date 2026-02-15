"""Tests para el Experimento Von Neumann — autómata celular elemental."""

import pytest
from experiments.von_neumann import VonNeumannExperiment


class TestVonNeumann:
    """Experimento Von Neumann: autómata celular 1D (Wolfram rules)."""

    def _run_single_row(self, rule: int, input_pattern: list[int]) -> list[int]:
        """Helper: corre un paso con un patrón de entrada y retorna la fila de salida."""
        width = len(input_pattern)
        exp = VonNeumannExperiment()
        exp.setup({"width": width, "height": 2, "rule": rule})

        # Activar patrón de entrada (fila inferior = fila height-1)
        for x, val in enumerate(input_pattern):
            if val == 1:
                exp.click(x, exp.height - 1)

        # Un step procesa la fila justo arriba de la entrada
        exp.step()
        frame = exp.get_frame()

        # Retorna fila 0 (salida)
        return frame[0]

    def test_rule_111_patron_conocido(self) -> None:
        """Rule 111 produce output correcto para patrón conocido."""
        # Rule 111 = 01101111 en binario
        # Patrón 111 → 0, todos los demás → 1
        # Con entrada [0, 0, 1, 0, 0], la celda central (idx=2) ve (0,1,0) → Rule bit 2 = 1
        # Celda 0 ve (0,0,0) con bordes → Rule bit 0 = 1
        # Celda 1 ve (0,0,1) → Rule bit 1 = 1
        # Celda 3 ve (1,0,0) → Rule bit 4 = 0
        # Celda 4 ve (0,0,0) con bordes → Rule bit 0 = 1
        output = self._run_single_row(111, [0, 0, 1, 0, 0])
        assert output == [1, 1, 1, 0, 1]

    def test_rule_30_patron_conocido(self) -> None:
        """Rule 30 produce output correcto para patrón conocido."""
        # Rule 30 = 00011110 en binario
        # 000→0, 001→1, 010→1, 011→1, 100→1, 101→0, 110→0, 111→0
        # Entrada: [0, 0, 1, 0, 0]
        # Celda 0: (0,0,0) → 0
        # Celda 1: (0,0,1) → 1
        # Celda 2: (0,1,0) → 1
        # Celda 3: (1,0,0) → 1
        # Celda 4: (0,0,0) → 0
        output = self._run_single_row(30, [0, 0, 1, 0, 0])
        assert output == [0, 1, 1, 1, 0]

    def test_celda_central_activada_triangulo(self) -> None:
        """Una celda central activada en la entrada produce el triángulo esperado (Rule 90)."""
        # Rule 90 = 01011010 en binario
        # 000→0, 001→1, 010→0, 011→1, 100→1, 101→0, 110→1, 111→0
        # Con una sola celda central, la primera generación debería producir:
        # Entrada:    [0, 0, 0, 0, 1, 0, 0, 0, 0]
        # Gen 1:      [0, 0, 0, 1, 0, 1, 0, 0, 0]
        exp = VonNeumannExperiment()
        exp.setup({"width": 9, "height": 3, "rule": 90})
        exp.click(4, 2)  # Centro de fila inferior
        exp.step()  # Procesa fila 1
        frame = exp.get_frame()
        assert frame[1] == [0, 0, 0, 1, 0, 1, 0, 0, 0]

    def test_experimento_orquesta_setup_click_step_get_frame(self) -> None:
        """El Experimento orquesta: setup → click → step → get_frame."""
        exp = VonNeumannExperiment()
        exp.setup({"width": 5, "height": 3, "rule": 111})

        # get_frame antes de click — todo ceros
        frame = exp.get_frame()
        assert all(cell == 0 for row in frame for cell in row)

        # click
        exp.click(2, 2)
        frame = exp.get_frame()
        assert frame[2][2] == 1  # La celda clickeada se activó

        # step
        exp.step()
        frame = exp.get_frame()
        # La fila 1 ya debería tener valores procesados
        assert any(cell != 0 for cell in frame[1])

    def test_reset_limpia_estado(self) -> None:
        """Reset reinicia el experimento."""
        exp = VonNeumannExperiment()
        exp.setup({"width": 5, "height": 3, "rule": 111})
        exp.click(2, 2)
        exp.step()

        exp.reset()
        frame = exp.get_frame()
        assert all(cell == 0 for row in frame for cell in row)

    def test_rule_110_patron_conocido(self) -> None:
        """Rule 110 produce output correcto para patrón conocido."""
        # Rule 110 = 01101110
        # 000→0, 001→1, 010→1, 011→1, 100→0, 101→1, 110→1, 111→0
        # Entrada: [0, 0, 1, 0, 0]
        # Celda 0: (0,0,0) → 0
        # Celda 1: (0,0,1) → 1
        # Celda 2: (0,1,0) → 1
        # Celda 3: (1,0,0) → 0
        # Celda 4: (0,0,0) → 0
        output = self._run_single_row(110, [0, 0, 1, 0, 0])
        assert output == [0, 1, 1, 0, 0]

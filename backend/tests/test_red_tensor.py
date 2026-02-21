"""Tests for RedTensor — vectorized parallel processing.

Verifies compilation, processing, set_valor, input masks, and get_grid.
"""

from __future__ import annotations

import random

import pytest
import torch

from core.constructor import Constructor
from core.neurona import Neurona, NeuronaEntrada
from core.red import Red
from core.constructor_tensor import ConstructorTensor
from core.masks import MASK_SIMPLE


def _crear_red_mexican_hat(width: int = 10, height: int = 10, seed: int = 42) -> Red:
    """Helper: creates a Red with Mexican hat mask and reproducible random values."""
    random.seed(seed)
    constructor = Constructor()
    red, _regiones = constructor.crear_grilla(
        width=width, height=height,
        filas_entrada=[], filas_salida=[],
        umbral=0.0,
    )
    constructor.aplicar_mascara_2d(red, width, height, MASK_SIMPLE)
    for neurona in red.neuronas:
        neurona.activar_external(random.random())
    return red


def _crear_red_von_neumann(width: int = 10, height: int = 10, regla: int = 110, seed: int = 42) -> Red:
    """Helper: creates a Von Neumann Red."""
    random.seed(seed)
    constructor = Constructor()
    fila_entrada = height - 1
    fila_salida = 0
    red, _regiones = constructor.crear_grilla(
        width=width, height=height,
        filas_entrada=[fila_entrada],
        filas_salida=[fila_salida],
        umbral=0.99,
    )
    for fila in range(height - 2, -1, -1):
        constructor.aplicar_regla_wolfram(
            red=red, regla=regla, fila_destino=fila, width=width, height=height,
        )
    for x in range(width):
        key = Constructor.key_by_coord(x, fila_entrada)
        neurona = red.get_neurona(key)
        if isinstance(neurona, NeuronaEntrada):
            neurona.activar_external(float(random.choice([0, 1])))
    return red


class TestRedTensorCompilacion:
    """ConstructorTensor.compilar preserves data correctly."""

    def test_compilar_preserva_valores(self):
        """compilar preserves the initial neuron values."""
        red = _crear_red_mexican_hat(10, 10)
        valores_antes = [n.valor for n in red.neuronas]

        red_tensor = ConstructorTensor.compilar(red)
        N = len(red.neuronas)
        valores_tensor = red_tensor.valores[:N].tolist()

        for i, (v_red, v_tensor) in enumerate(zip(valores_antes, valores_tensor)):
            assert abs(v_red - v_tensor) < 1e-6, (
                f"Neurona {i}: red={v_red}, tensor={v_tensor}"
            )

    def test_procesar_n_equivale_a_n_steps(self):
        """procesar_n(5) gives the same result as 5 calls to procesar()."""
        red_a = _crear_red_mexican_hat(8, 8, seed=77)
        red_b = _crear_red_mexican_hat(8, 8, seed=77)
        tensor_a = ConstructorTensor.compilar(red_a)
        tensor_b = ConstructorTensor.compilar(red_b)

        for _ in range(5):
            tensor_a.procesar()
        tensor_b.procesar_n(5)

        N = len(red_a.neuronas)
        for i in range(N):
            va = tensor_a.valores[i].item()
            vb = tensor_b.valores[i].item()
            assert abs(va - vb) < 1e-6, (
                f"Neurona {i}: procesar()×5={va}, procesar_n(5)={vb}"
            )


class TestRedTensorSetValor:
    """set_valor modifies the tensor correctly."""

    def test_set_valor_modifica_neurona(self):
        """set_valor changes the value of the specified neuron."""
        red = _crear_red_mexican_hat(5, 5, seed=1)
        red_tensor = ConstructorTensor.compilar(red)

        red_tensor.set_valor(0, 1.0)
        assert red_tensor.valores[0].item() == 1.0

        red_tensor.set_valor(0, 0.0)
        assert red_tensor.valores[0].item() == 0.0

    def test_set_valor_no_afecta_otras_neuronas(self):
        """set_valor only modifies the specified neuron."""
        red = _crear_red_mexican_hat(5, 5, seed=1)
        red_tensor = ConstructorTensor.compilar(red)

        valores_antes = red_tensor.valores.clone()
        red_tensor.set_valor(3, 0.999)

        for i in range(red_tensor.N):
            if i == 3:
                assert red_tensor.valores[i].item() == pytest.approx(0.999, abs=1e-5)
            else:
                if i < len(valores_antes):
                    assert red_tensor.valores[i].item() == valores_antes[i].item()


class TestRedTensorMascaraEntrada:
    """NeuronaEntrada is not modified during procesar()."""

    def test_mascara_entrada_preserva_valores(self):
        """NeuronaEntrada neurons maintain their value after procesar()."""
        red = _crear_red_von_neumann(10, 10, seed=42)
        red_tensor = ConstructorTensor.compilar(red)

        # Get indices of NeuronaEntrada
        entrada_indices = []
        for i, neurona in enumerate(red.neuronas):
            if isinstance(neurona, NeuronaEntrada):
                entrada_indices.append(i)

        # Record their values
        valores_entrada_antes = {
            i: red_tensor.valores[i].item() for i in entrada_indices
        }

        # Process
        red_tensor.procesar()

        # Verify they haven't changed
        for i in entrada_indices:
            assert red_tensor.valores[i].item() == valores_entrada_antes[i], (
                f"NeuronaEntrada {i} changed from {valores_entrada_antes[i]} "
                f"to {red_tensor.valores[i].item()}"
            )


class TestRedTensorGetGrid:
    """get_grid returns the grid with correct dimensions."""

    def test_get_grid_dimensiones(self):
        """get_grid returns height rows × width columns."""
        red = _crear_red_mexican_hat(8, 6, seed=1)
        red_tensor = ConstructorTensor.compilar(red)

        grid = red_tensor.get_grid(8, 6)
        assert len(grid) == 6
        for row in grid:
            assert len(row) == 8

    def test_get_grid_valores_correctos(self):
        """get_grid returns the same values as Red.get_grid."""
        red = _crear_red_mexican_hat(8, 6, seed=99)
        red2 = _crear_red_mexican_hat(8, 6, seed=99)
        red_tensor = ConstructorTensor.compilar(red2)

        grid_red = red.get_grid(8, 6)
        grid_tensor = red_tensor.get_grid(8, 6)

        for y in range(6):
            for x in range(8):
                assert abs(grid_red[y][x] - grid_tensor[y][x]) < 1e-5, (
                    f"({x},{y}): red={grid_red[y][x]}, tensor={grid_tensor[y][x]}"
                )


class TestRedTensorBalanceado:
    """RedTensor with balanced weights works correctly."""

    def test_balanceado_procesa(self):
        """Red with balancing produces binary values after one step."""
        random.seed(55)
        constructor = Constructor()
        red, _ = constructor.crear_grilla(
            width=10, height=10,
            filas_entrada=[], filas_salida=[],
            umbral=0.0,
        )
        constructor.aplicar_mascara_2d(red, 10, 10, MASK_SIMPLE)
        constructor.balancear_pesos(list(red.neuronas), target=0.0)
        for n in red.neuronas:
            n.activar_external(random.random())

        red_tensor = ConstructorTensor.compilar(red)
        red_tensor.procesar()

        N = len(red.neuronas)
        for i in range(N):
            v = red_tensor.valores[i].item()
            assert v == 0.0 or v == 1.0, f"Neurona {i}: valor={v} (expected 0 or 1)"

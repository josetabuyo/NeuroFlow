"""Tests de equivalencia: RedTensor debe producir los mismos resultados que Red.

Estos tests verifican que el motor tensorial PyTorch produce exactamente
los mismos valores que el motor secuencial legacy para los mismos datos.
"""

from __future__ import annotations

import random

import pytest
import torch

from core.constructor import Constructor
from core.neurona import Neurona, NeuronaEntrada
from core.red import Red
from core.constructor_tensor import ConstructorTensor
from experiments.kohonen import KOHONEN_SIMPLE_MASK


def _crear_red_kohonen(width: int = 10, height: int = 10, seed: int = 42) -> Red:
    """Helper: crea una Red Kohonen con valores aleatorios reproducibles."""
    random.seed(seed)
    constructor = Constructor()
    red, _regiones = constructor.crear_grilla(
        width=width, height=height,
        filas_entrada=[], filas_salida=[],
        umbral=0.0,
    )
    constructor.aplicar_mascara_2d(red, width, height, KOHONEN_SIMPLE_MASK)
    for neurona in red.neuronas:
        neurona.activar_external(random.random())
    return red


def _crear_red_von_neumann(width: int = 10, height: int = 10, regla: int = 110, seed: int = 42) -> Red:
    """Helper: crea una Red Von Neumann."""
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
            red=red, regla=regla, fila_destino=fila, width=width,
        )
    # Activate some input neurons
    for x in range(width):
        key = Constructor.key_by_coord(x, fila_entrada)
        neurona = red.get_neurona(key)
        if isinstance(neurona, NeuronaEntrada):
            neurona.activar_external(float(random.choice([0, 1])))
    return red


def _get_all_values(red: Red) -> list[float]:
    """Extract all neuron values from a Red in order."""
    return [n.valor for n in red.neuronas]


class TestRedTensorEquivalencia:
    """RedTensor debe producir los mismos valores que Red secuencial."""

    def test_compilar_preserva_valores(self):
        """ConstructorTensor.compilar preserva los valores iniciales."""
        red = _crear_red_kohonen(10, 10)
        valores_antes = _get_all_values(red)

        red_tensor = ConstructorTensor.compilar(red)
        # RedTensor may have an extra zero neuron, compare only first N
        N = len(red.neuronas)
        valores_tensor = red_tensor.valores[:N].tolist()

        for i, (v_red, v_tensor) in enumerate(zip(valores_antes, valores_tensor)):
            assert abs(v_red - v_tensor) < 1e-6, (
                f"Neurona {i}: red={v_red}, tensor={v_tensor}"
            )

    def test_un_step_kohonen_equivalente(self):
        """Un step de Kohonen produce los mismos resultados en Red y RedTensor."""
        # Create two identical networks with same seed
        red = _crear_red_kohonen(10, 10, seed=42)
        red2 = _crear_red_kohonen(10, 10, seed=42)
        red_tensor = ConstructorTensor.compilar(red2)

        # Process one step each
        red.procesar()
        red_tensor.procesar()

        # Compare values
        N = len(red.neuronas)
        for i in range(N):
            v_red = red.neuronas[i].valor
            v_tensor = red_tensor.valores[i].item()
            assert abs(v_red - v_tensor) < 1e-5, (
                f"Neurona {i} ({red.neuronas[i].id}): "
                f"red={v_red}, tensor={v_tensor}"
            )

    def test_multiples_steps_kohonen_equivalente(self):
        """5 steps de Kohonen producen los mismos resultados."""
        red = _crear_red_kohonen(10, 10, seed=123)
        red2 = _crear_red_kohonen(10, 10, seed=123)
        red_tensor = ConstructorTensor.compilar(red2)

        for step_num in range(5):
            red.procesar()
            red_tensor.procesar()

            N = len(red.neuronas)
            for i in range(N):
                v_red = red.neuronas[i].valor
                v_tensor = red_tensor.valores[i].item()
                assert abs(v_red - v_tensor) < 1e-5, (
                    f"Step {step_num}, Neurona {i}: "
                    f"red={v_red}, tensor={v_tensor}"
                )

    def test_von_neumann_un_step_equivalente(self):
        """Un step de Von Neumann (procesamiento por fila) equivalente.

        Von Neumann procesa fila por fila, no toda la red. Para equivalencia
        completa, procesamos la red entera (como hace Kohonen) y comparamos.
        """
        red = _crear_red_von_neumann(10, 10, regla=110, seed=42)
        red2 = _crear_red_von_neumann(10, 10, regla=110, seed=42)
        red_tensor = ConstructorTensor.compilar(red2)

        # Process full network
        red.procesar()
        red_tensor.procesar()

        N = len(red.neuronas)
        for i in range(N):
            v_red = red.neuronas[i].valor
            v_tensor = red_tensor.valores[i].item()
            assert abs(v_red - v_tensor) < 1e-5, (
                f"Neurona {i} ({red.neuronas[i].id}): "
                f"red={v_red}, tensor={v_tensor}"
            )

    def test_procesar_n_equivale_a_n_steps(self):
        """procesar_n(5) da lo mismo que 5 llamadas a procesar()."""
        red_a = _crear_red_kohonen(8, 8, seed=77)
        red_b = _crear_red_kohonen(8, 8, seed=77)
        tensor_a = ConstructorTensor.compilar(red_a)
        tensor_b = ConstructorTensor.compilar(red_b)

        # tensor_a: 5 calls to procesar()
        for _ in range(5):
            tensor_a.procesar()

        # tensor_b: one call to procesar_n(5)
        tensor_b.procesar_n(5)

        N = len(red_a.neuronas)
        for i in range(N):
            va = tensor_a.valores[i].item()
            vb = tensor_b.valores[i].item()
            assert abs(va - vb) < 1e-6, (
                f"Neurona {i}: procesar()×5={va}, procesar_n(5)={vb}"
            )


class TestRedTensorSetValor:
    """set_valor modifica el tensor correctamente."""

    def test_set_valor_modifica_neurona(self):
        """set_valor cambia el valor de la neurona indicada."""
        red = _crear_red_kohonen(5, 5, seed=1)
        red_tensor = ConstructorTensor.compilar(red)

        red_tensor.set_valor(0, 1.0)
        assert red_tensor.valores[0].item() == 1.0

        red_tensor.set_valor(0, 0.0)
        assert red_tensor.valores[0].item() == 0.0

    def test_set_valor_no_afecta_otras_neuronas(self):
        """set_valor solo modifica la neurona indicada."""
        red = _crear_red_kohonen(5, 5, seed=1)
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
    """NeuronaEntrada no se modifica durante procesar()."""

    def test_mascara_entrada_preserva_valores(self):
        """Las NeuronaEntrada mantienen su valor después de procesar()."""
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
                f"NeuronaEntrada {i} cambió de {valores_entrada_antes[i]} "
                f"a {red_tensor.valores[i].item()}"
            )


class TestRedTensorGetGrid:
    """get_grid retorna la grilla con las dimensiones correctas."""

    def test_get_grid_dimensiones(self):
        """get_grid retorna height filas × width columnas."""
        red = _crear_red_kohonen(8, 6, seed=1)
        red_tensor = ConstructorTensor.compilar(red)

        grid = red_tensor.get_grid(8, 6)
        assert len(grid) == 6
        for row in grid:
            assert len(row) == 8

    def test_get_grid_valores_correctos(self):
        """get_grid retorna los mismos valores que Red.get_grid."""
        red = _crear_red_kohonen(8, 6, seed=99)
        red2 = _crear_red_kohonen(8, 6, seed=99)
        red_tensor = ConstructorTensor.compilar(red2)

        grid_red = red.get_grid(8, 6)
        grid_tensor = red_tensor.get_grid(8, 6)

        for y in range(6):
            for x in range(8):
                assert abs(grid_red[y][x] - grid_tensor[y][x]) < 1e-5, (
                    f"({x},{y}): red={grid_red[y][x]}, tensor={grid_tensor[y][x]}"
                )


class TestRedTensorKohonenBalanceado:
    """Equivalencia con Kohonen Balanceado (pesos ajustados)."""

    def test_kohonen_balanceado_equivalente(self):
        """Kohonen con balanceo produce los mismos resultados."""
        random.seed(55)
        constructor = Constructor()
        red, _ = constructor.crear_grilla(
            width=10, height=10,
            filas_entrada=[], filas_salida=[],
            umbral=0.0,
        )
        constructor.aplicar_mascara_2d(red, 10, 10, KOHONEN_SIMPLE_MASK)
        constructor.balancear_pesos(list(red.neuronas), target=0.0)
        for n in red.neuronas:
            n.activar_external(random.random())

        # Create identical copy
        random.seed(55)
        red2, _ = constructor.crear_grilla(
            width=10, height=10,
            filas_entrada=[], filas_salida=[],
            umbral=0.0,
        )
        constructor.aplicar_mascara_2d(red2, 10, 10, KOHONEN_SIMPLE_MASK)
        constructor.balancear_pesos(list(red2.neuronas), target=0.0)
        for n in red2.neuronas:
            n.activar_external(random.random())

        red_tensor = ConstructorTensor.compilar(red2)

        # Process 3 steps
        for step_num in range(3):
            red.procesar()
            red_tensor.procesar()

            N = len(red.neuronas)
            for i in range(N):
                v_red = red.neuronas[i].valor
                v_tensor = red_tensor.valores[i].item()
                assert abs(v_red - v_tensor) < 1e-5, (
                    f"Step {step_num}, Neurona {i}: "
                    f"red={v_red}, tensor={v_tensor}"
                )

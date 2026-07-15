"""Tests for BrainTensor — vectorized parallel processing.

Verifies compilation, processing, set_valor, input masks, and get_grid.
"""

from __future__ import annotations

import random

import pytest
import torch

from core.constructor import Constructor
from core.neurona import Neurona, NeuronaEntrada
from core.brain import Brain
from core.constructor_tensor import ConstructorTensor
from core.masks import MASK_SIMPLE


def _crear_brain_mexican_hat(width: int = 10, height: int = 10, seed: int = 42) -> Brain:
    """Helper: creates a Brain with Mexican hat mask and reproducible random values."""
    random.seed(seed)
    constructor = Constructor()
    brain, _regiones = constructor.crear_grilla(
        width=width, height=height,
        filas_entrada=[], filas_salida=[],
        umbral=0.0,
    )
    constructor.aplicar_mascara_2d(brain, width, height, MASK_SIMPLE)
    for neurona in brain.neuronas:
        neurona.activar_external(random.random())
    return brain


def _crear_brain_von_neumann(width: int = 10, height: int = 10, regla: int = 110, seed: int = 42) -> Brain:
    """Helper: creates a Von Neumann Brain."""
    random.seed(seed)
    constructor = Constructor()
    fila_entrada = height - 1
    fila_salida = 0
    brain, _regiones = constructor.crear_grilla(
        width=width, height=height,
        filas_entrada=[fila_entrada],
        filas_salida=[fila_salida],
        umbral=0.99,
    )
    for fila in range(height - 2, -1, -1):
        constructor.aplicar_regla_wolfram(
            brain=brain, regla=regla, fila_destino=fila, width=width, height=height,
        )
    for x in range(width):
        key = Constructor.key_by_coord(x, fila_entrada)
        neurona = brain.get_neurona(key)
        if isinstance(neurona, NeuronaEntrada):
            neurona.activar_external(float(random.choice([0, 1])))
    return brain


class TestBrainTensorCompilacion:
    """ConstructorTensor.compilar preserves data correctly."""

    def test_compilar_preserva_valores(self):
        """compilar preserves the initial neuron values."""
        brain = _crear_brain_mexican_hat(10, 10)
        valores_antes = [n.valor for n in brain.neuronas]

        brain_tensor = ConstructorTensor.compilar(brain)
        N = len(brain.neuronas)
        valores_tensor = brain_tensor.valores[:N].tolist()

        for i, (v_brain, v_tensor) in enumerate(zip(valores_antes, valores_tensor)):
            assert abs(v_brain - v_tensor) < 1e-6, (
                f"Neurona {i}: brain={v_brain}, tensor={v_tensor}"
            )

    def test_procesar_n_equivale_a_n_steps(self):
        """procesar_n(5) gives the same result as 5 calls to procesar()."""
        brain_a = _crear_brain_mexican_hat(8, 8, seed=77)
        brain_b = _crear_brain_mexican_hat(8, 8, seed=77)
        tensor_a = ConstructorTensor.compilar(brain_a)
        tensor_b = ConstructorTensor.compilar(brain_b)

        for _ in range(5):
            tensor_a.procesar()
        tensor_b.procesar_n(5)

        N = len(brain_a.neuronas)
        for i in range(N):
            va = tensor_a.valores[i].item()
            vb = tensor_b.valores[i].item()
            assert abs(va - vb) < 1e-6, (
                f"Neurona {i}: procesar()×5={va}, procesar_n(5)={vb}"
            )


def _bare_brain_tensor(pesos: list[list[float]], valid: list[list[bool]], grupo_ids: list[list[int]]) -> "BrainTensor":
    """A minimal BrainTensor with only what _compute_tension's group_avg
    branch reads: per-dendrite weight, validity, and declared group index.
    One row = one neuron; bypasses Constructor/ConstructorTensor entirely so
    the group-then-sign averaging math can be tested in isolation."""
    from core.brain_tensor import BrainTensor
    bt = object.__new__(BrainTensor)
    bt.device = "cpu"
    bt._dend_pesos = torch.tensor(pesos, dtype=torch.float32)
    bt._dendrita_mascara = torch.tensor(valid, dtype=torch.bool)
    bt._dendrite_grupo_ids = torch.tensor(grupo_ids, dtype=torch.int64)
    bt.max_grupos = int(bt._dendrite_grupo_ids.max().item()) + 1
    return bt


class TestGroupAvgTension:
    """_compute_tension('group_avg') averages dendrites within a declared
    group first, then averages present groups by sign, then diffs — one vote
    per group, not per raw dendrite."""

    def test_single_group_per_sign_matches_flat_average(self):
        """No divergence case: one positive group (2 dendrites) and one
        negative group (1 dendrite) — same as flat per-sign averaging."""
        bt = _bare_brain_tensor(
            pesos=[[0.6, 0.6, -0.7]],
            valid=[[True, True, True]],
            grupo_ids=[[0, 0, 1]],
        )
        dpc = torch.tensor([[0.6, 0.6, -0.7]])
        tension = bt._compute_tension(dpc, "group_avg", 0, 1)
        assert tension.item() == pytest.approx(0.6 + (-0.7))

    def test_two_positive_groups_vote_equally_regardless_of_dendrite_count(self):
        """The divergence case: group A has 1 dendrite (value 0.6), group B
        has 3 dendrites (all 0.2) — a flat per-dendrite average would skew
        toward B (4 dendrites total, 3 of them 0.2); group-then-sign gives
        each GROUP one vote: mean(0.6, 0.2) = 0.4, not mean of the 4 raw
        values (0.6+0.2*3)/4 = 0.3."""
        bt = _bare_brain_tensor(
            pesos=[[0.6, 0.2, 0.2, 0.2]],
            valid=[[True, True, True, True]],
            grupo_ids=[[0, 1, 1, 1]],
        )
        dpc = torch.tensor([[0.6, 0.2, 0.2, 0.2]])
        tension = bt._compute_tension(dpc, "group_avg", 0, 1)
        assert tension.item() == pytest.approx(0.4)

    def test_absent_group_excluded_not_zero_padded(self):
        """A neuron with only a negative group (no positive dendrites at all)
        gets neg_avg alone — the missing positive side must not pull the
        result toward 0 as if it contributed a 0-valued group."""
        bt = _bare_brain_tensor(
            pesos=[[-0.7, -0.7]],
            valid=[[True, True]],
            grupo_ids=[[0, 0]],
        )
        dpc = torch.tensor([[-0.7, -0.7]])
        tension = bt._compute_tension(dpc, "group_avg", 0, 1)
        assert tension.item() == pytest.approx(-0.7)

    def test_invalid_dendrite_slots_excluded(self):
        """Padding/invalid dendrite slots (mask False) must not count toward
        any group's average."""
        bt = _bare_brain_tensor(
            pesos=[[0.6, 0.6, 0.0]],
            valid=[[True, True, False]],
            grupo_ids=[[0, 0, 0]],
        )
        dpc = torch.tensor([[0.6, 0.6, 0.0]])
        tension = bt._compute_tension(dpc, "group_avg", 0, 1)
        assert tension.item() == pytest.approx(0.6)


class TestBrainTensorSetValor:
    """set_valor modifies the tensor correctly."""

    def test_set_valor_modifica_neurona(self):
        """set_valor changes the value of the specified neuron."""
        brain = _crear_brain_mexican_hat(5, 5, seed=1)
        brain_tensor = ConstructorTensor.compilar(brain)

        brain_tensor.set_valor(0, 1.0)
        assert brain_tensor.valores[0].item() == 1.0

        brain_tensor.set_valor(0, 0.0)
        assert brain_tensor.valores[0].item() == 0.0

    def test_set_valor_no_afecta_otras_neuronas(self):
        """set_valor only modifies the specified neuron."""
        brain = _crear_brain_mexican_hat(5, 5, seed=1)
        brain_tensor = ConstructorTensor.compilar(brain)

        valores_antes = brain_tensor.valores.clone()
        brain_tensor.set_valor(3, 0.999)

        for i in range(brain_tensor.N):
            if i == 3:
                assert brain_tensor.valores[i].item() == pytest.approx(0.999, abs=1e-5)
            else:
                if i < len(valores_antes):
                    assert brain_tensor.valores[i].item() == valores_antes[i].item()


class TestBrainTensorMascaraEntrada:
    """NeuronaEntrada is not modified during procesar()."""

    def test_mascara_entrada_preserva_valores(self):
        """NeuronaEntrada neurons maintain their value after procesar()."""
        brain = _crear_brain_von_neumann(10, 10, seed=42)
        brain_tensor = ConstructorTensor.compilar(brain)

        # Get indices of NeuronaEntrada
        entrada_indices = []
        for i, neurona in enumerate(brain.neuronas):
            if isinstance(neurona, NeuronaEntrada):
                entrada_indices.append(i)

        # Record their values
        valores_entrada_antes = {
            i: brain_tensor.valores[i].item() for i in entrada_indices
        }

        # Process
        brain_tensor.procesar()

        # Verify they haven't changed
        for i in entrada_indices:
            assert brain_tensor.valores[i].item() == valores_entrada_antes[i], (
                f"NeuronaEntrada {i} changed from {valores_entrada_antes[i]} "
                f"to {brain_tensor.valores[i].item()}"
            )


class TestBrainTensorGetGrid:
    """get_grid returns the grid with correct dimensions."""

    def test_get_grid_dimensiones(self):
        """get_grid returns height rows × width columns."""
        brain = _crear_brain_mexican_hat(8, 6, seed=1)
        brain_tensor = ConstructorTensor.compilar(brain)

        grid = brain_tensor.get_grid(8, 6)
        assert len(grid) == 6
        for row in grid:
            assert len(row) == 8

    def test_get_grid_valores_correctos(self):
        """get_grid returns the same values as Brain.get_grid."""
        brain = _crear_brain_mexican_hat(8, 6, seed=99)
        brain2 = _crear_brain_mexican_hat(8, 6, seed=99)
        brain_tensor = ConstructorTensor.compilar(brain2)

        grid_brain = brain.get_grid(8, 6)
        grid_tensor = brain_tensor.get_grid(8, 6)

        for y in range(6):
            for x in range(8):
                assert abs(grid_brain[y][x] - grid_tensor[y][x]) < 1e-5, (
                    f"({x},{y}): brain={grid_brain[y][x]}, tensor={grid_tensor[y][x]}"
                )


class TestBrainTensorProcessing:
    """BrainTensor produces correct binary values after processing."""

    def test_procesar_produces_binary_values(self):
        """Brain produces binary values after one step."""
        random.seed(55)
        constructor = Constructor()
        brain, _ = constructor.crear_grilla(
            width=10, height=10,
            filas_entrada=[], filas_salida=[],
            umbral=0.0,
        )
        constructor.aplicar_mascara_2d(brain, 10, 10, MASK_SIMPLE)
        for n in brain.neuronas:
            n.activar_external(random.random())

        brain_tensor = ConstructorTensor.compilar(brain)
        brain_tensor.procesar()

        N = len(brain.neuronas)
        for i in range(N):
            v = brain_tensor.valores[i].item()
            assert v == 0.0 or v == 1.0, f"Neurona {i}: valor={v} (expected 0 or 1)"


class TestSoftActivation:
    """Neurons in soft regions output tension clamped to [0,1], not binary."""

    def test_soft_produces_analog_values(self):
        """With activation=soft all neurons output non-binary values in [0,1]."""
        random.seed(42)
        constructor = Constructor()
        brain, _ = constructor.crear_grilla(
            width=6, height=6, filas_entrada=[], filas_salida=[], umbral=0.5,
        )
        constructor.aplicar_mascara_2d(brain, 6, 6, MASK_SIMPLE)
        for n in brain.neuronas:
            n.activar_external(random.random())

        N = 36
        region_specs = [(0, N, "min_vs_max", [], True)]
        bt = ConstructorTensor.compilar(brain, region_specs=region_specs)
        bt.procesar()

        vals = bt.valores[:N].tolist()
        assert all(0.0 <= v <= 1.0 for v in vals), "Soft values must stay in [0,1]"
        non_binary = [v for v in vals if 0.001 < v < 0.999]
        assert len(non_binary) > 0, "Soft activation must produce analog (non-0/1) values"

    def test_soft_preserves_input_neurons(self):
        """NeuronaEntrada values are preserved even when their region is soft."""
        random.seed(7)
        constructor = Constructor()
        brain, _ = constructor.crear_grilla(
            width=4, height=4, filas_entrada=[3], filas_salida=[], umbral=0.0,
        )
        constructor.aplicar_mascara_2d(brain, 4, 4, MASK_SIMPLE)
        for n in brain.neuronas:
            n.activar_external(0.7)

        N = 16
        region_specs = [(0, N, "min_vs_max", [], True)]
        bt = ConstructorTensor.compilar(brain, region_specs=region_specs)
        for i in range(12, 16):
            bt.valores[i] = 0.3
        bt.procesar()

        for i in range(12, 16):
            assert abs(bt.valores[i].item() - 0.3) < 1e-5, f"Input neuron {i} must be preserved"

    def test_binary_region_still_binary_alongside_soft(self):
        """Non-soft regions produce binary 0/1 even when other regions are soft."""
        random.seed(13)
        constructor = Constructor()
        brain, _ = constructor.crear_grilla(
            width=4, height=4, filas_entrada=[], filas_salida=[], umbral=0.0,
        )
        constructor.aplicar_mascara_2d(brain, 4, 4, MASK_SIMPLE)
        for n in brain.neuronas:
            n.activar_external(random.random())

        N = 16
        # First half binary, second half soft (no input neurons in this brain)
        region_specs = [(0, 8, "min_vs_max", [], False), (8, N, "min_vs_max", [], True)]
        bt = ConstructorTensor.compilar(brain, region_specs=region_specs)
        bt.procesar()

        # First 8 neurons: binary
        for i in range(8):
            v = bt.valores[i].item()
            assert v == 0.0 or v == 1.0, f"Binary neuron {i} should be 0 or 1, got {v}"

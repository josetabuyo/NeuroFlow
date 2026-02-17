"""Benchmark: Motor secuencial (legacy) vs Motor tensorial (PyTorch).

Compara la performance de Red.procesar() vs RedTensor.procesar()
para diferentes tamaños de grilla con máscara Kohonen.

Ejecutar:
    cd backend
    python benchmark.py
"""

from __future__ import annotations

import random
import time

from core.constructor import Constructor
from core.red import Red
from core.constructor_tensor import ConstructorTensor
from experiments.kohonen import KOHONEN_SIMPLE_MASK


def crear_red_kohonen(width: int, height: int, seed: int = 42) -> Red:
    """Crea una Red Kohonen con valores aleatorios reproducibles."""
    random.seed(seed)
    constructor = Constructor()
    red, _ = constructor.crear_grilla(
        width=width, height=height,
        filas_entrada=[], filas_salida=[],
        umbral=0.0,
    )
    constructor.aplicar_mascara_2d(red, width, height, KOHONEN_SIMPLE_MASK)
    for neurona in red.neuronas:
        neurona.activar_external(random.random())
    return red


def benchmark_secuencial(red: Red, steps: int) -> float:
    """Benchmarks sequential Red.procesar()."""
    t0 = time.perf_counter()
    for _ in range(steps):
        red.procesar()
    return time.perf_counter() - t0


def benchmark_tensor(red: Red, steps: int) -> float:
    """Benchmarks tensor RedTensor.procesar_n()."""
    red_tensor = ConstructorTensor.compilar(red)
    t0 = time.perf_counter()
    red_tensor.procesar_n(steps)
    return time.perf_counter() - t0


def main() -> None:
    SIZES = [
        (10, 10),     # 100 neuronas — micro
        (30, 30),     # 900 neuronas — baseline actual
        (50, 50),     # 2.500 neuronas
        (100, 100),   # 10.000 neuronas
    ]
    STEPS = 100

    print("=" * 70)
    print("  BENCHMARK: Motor Secuencial vs Motor Tensorial PyTorch")
    print("=" * 70)
    print(f"  Steps per size: {STEPS}")
    print(f"  Mask: kohonen_simple (9 dendritas, ~80 sinapsis/neurona)")
    print()
    print(f"{'Size':>10} {'Neuronas':>10} {'Secuencial':>12} {'Tensor':>12} {'Speedup':>10}")
    print("-" * 70)

    for w, h in SIZES:
        n = w * h

        # Create two identical networks
        red_seq = crear_red_kohonen(w, h, seed=42)
        red_ten = crear_red_kohonen(w, h, seed=42)

        # Benchmark sequential
        t_seq = benchmark_secuencial(red_seq, STEPS)

        # Benchmark tensor
        t_ten = benchmark_tensor(red_ten, STEPS)

        speedup = t_seq / t_ten if t_ten > 0 else float("inf")
        print(f"{w:>4}×{h:<4} {n:>10,} {t_seq:>11.3f}s {t_ten:>11.3f}s {speedup:>9.1f}×")

    print("-" * 70)
    print()
    print("  Si speedup > 10× para 30×30, el motor tensorial vale la pena.")
    print("  Proceder a Fase 2: eliminar legacy.")
    print()


if __name__ == "__main__":
    main()

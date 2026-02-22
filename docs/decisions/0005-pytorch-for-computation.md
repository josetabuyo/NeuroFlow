# 0005 — PyTorch for Neural Computation

**Status:** Accepted

**Date:** 2025

## Context

The neural model requires per-step computation across all neurons:
synapse recognition (`1 - |peso - input|`), dendrite aggregation
(average × weight), and neuron activation (fuzzy OR + threshold). With
2500 neurons (50×50), each having dozens of synapses, pure Python loops
are too slow for real-time frame rates.

## Decision

Use PyTorch tensors for all neural computation. The OOP graph
(Red → Neurona → Dendrita → Sinapsis) is compiled into flat tensors
by `ConstructorTensor`, and `RedTensor.procesar()` executes all
formulas as vectorized operations.

## Consequences

- ~100x speedup over Python loops — instant processing for 50×50 grids.
- Scales to larger grids without architectural changes.
- GPU acceleration (`cuda`) available when needed, with no code changes.
- Rich scatter/gather operations (e.g. `scatter_reduce`) fit the
  synapse→dendrite→neuron aggregation pattern naturally.
- Trade-off: PyTorch is a heavy dependency (~2GB) for what is
  conceptually simple math. NumPy could suffice at smaller scales but
  lacks GPU support and some gather operations.

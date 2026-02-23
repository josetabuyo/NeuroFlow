# Technical Architecture

Technical design of the system: stack, classes, API, protocol, and hosting.

For project vision and philosophy, see [Vision](VISION.md).
For the roadmap, see [Stages](STAGES.md).
For the neural model close to the code, see [Neural Model](../backend/core/README.md).

---

## 1. Technology Stack

```
┌─────────────────────────────────────────────────────────┐
│                      FRONTEND                           │
│   Vite + React + TypeScript + HTML5 Canvas              │
│   Deploy: Vercel (free)                                 │
└──────────────────────┬──────────────────────────────────┘
                       │ WebSocket (ws://)
                       │ real-time frames
┌──────────────────────┴──────────────────────────────────┐
│                      BACKEND                            │
│   Python 3.11+ / FastAPI / uvicorn                      │
│   PyTorch for tensor operations                         │
│   Deploy: Render.com (free, 750h/month)                  │
└─────────────────────────────────────────────────────────┘
```

### Rationale

| Component | Choice | Why |
|-----------|--------|-----|
| Backend framework | **FastAPI** | Native async, WebSocket, typing, most popular in Python 2025-2026 |
| Backend runtime | **uvicorn** | Standard ASGI server for FastAPI |
| Compute | **PyTorch** | Vectorized tensor operations, GPU-ready |
| Frontend bundler | **Vite** | Instant build, HMR, current standard |
| Frontend framework | **React 19 + TypeScript** | Most adopted, reusable components |
| Rendering | **HTML5 Canvas** | Direct, fast, perfect for pixel grids |
| Communication | **WebSocket** | Bidirectional, low latency, ideal for frame streaming |
| Backend hosting | **Render.com** | Only one with real free tier for Python (750h/month) |
| Frontend hosting | **Vercel** | Generous free tier, auto-deploy from Git, optimal for React |
| Backend tests | **pytest** | Python standard, simple, powerful |
| Frontend tests (E2E) | **Playwright** | Browser automation, headless and interactive modes |

---

## 2. Project Structure

```
NeuroFlow/
├── backend/
│   ├── main.py                    # FastAPI entry point
│   ├── requirements.txt           # Python dependencies
│   ├── core/                      # Neural model (port from RedJavaScript)
│   │   ├── __init__.py
│   │   ├── sinapsis.py            # Synaptic connection
│   │   ├── dendrita.py            # Dendritic branch
│   │   ├── neurona.py             # Neuron + NeuronaEntrada
│   │   ├── brain.py               # Neural network (dumb container)
│   │   ├── region.py              # Neuron grouping (organization)
│   │   └── constructor.py        # Factory/builder for networks and regions
│   ├── experiments/               # Experiments (plug-in)
│   │   ├── __init__.py
│   │   ├── base.py                # Base Experiment class
│   │   └── deamons_lab.py         # Deamons Lab (wiring laboratory)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── websocket.py           # WebSocket handler
│   │   └── routes.py              # REST endpoints
│   └── tests/
│       ├── conftest.py
│       ├── test_sinapsis.py
│       ├── test_dendrita.py
│       ├── test_neurona.py
│       ├── test_brain.py           # Brain does NOT know about regions
│       ├── test_region.py          # Region is just grouping
│       ├── test_constructor.py     # Constructor assembles Brain + Regions
│       ├── test_deamons_lab.py
│       └── test_brain_tensor.py
│
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx               # React entry point
│       ├── App.tsx                # Main layout
│       ├── components/
│       │   ├── PixelCanvas.tsx    # Grid rendering
│       │   ├── Sidebar.tsx        # Experiments panel
│       │   └── Controls.tsx       # Play/Pause/Step/Reset
│       ├── hooks/
│       │   └── useExperiment.ts   # WebSocket + experiment state
│       └── types/
│           └── index.ts           # Shared types
│
├── docs/
│   ├── ARCHITECTURE.md            # This document
│   ├── VISION.md                  # Philosophy, daemons, mind model
│   ├── STAGES.md                  # Roadmap (5 stages)
│   ├── REFERENCES.md              # Complete bibliography
│   ├── AUTHOR.md                  # About the author and dedication
│   └── decisions/                 # Architecture Decision Records (ADR)
│       ├── README.md              # Index of all decisions
│       ├── 0001-web-app-not-notebooks.md
│       ├── 0002-separate-frontend-backend.md
│       ├── 0003-websocket-not-polling.md
│       ├── 0004-react-over-svelte-vue.md
│       └── 0005-pytorch-for-computation.md
│
├── README.md                      # Entry point, navigation
└── .gitignore
```

---

## 3. Neural Model (Core)

Port of the original model to Python, with a key architectural improvement:
**separation of responsibilities between processing and organization**.

### 3.0 Design Principle: Separation of Responsibilities

In the original project (RedJavaScript), the `Brain` class knew about regions
(INPUT, OUTPUT, INTERNAL) and decided which neurons to process. This couples
organization with processing.

In NeuroFlow we separate these responsibilities:

```
PROCESSING (does not know about organization)     ORGANIZATION (does not know about processing)
┌──────────────────────────────┐                  ┌──────────────────────────────┐
│  Brain                       │                  │  Region                      │
│  Only contains neurons.      │                  │  Named group of neurons.    │
│  Only processes all of them.│                  │  References only, not owner.│
│  Does not know input/output. │                  │  Useful for connecting,     │
│                              │                  │  activating and reading     │
│  Sinapsis → Dendrita →       │                  │  subsets.                   │
│  Neurona                     │                  │                              │
│  (each knows how to process) │                  │  Constructor                │
│                              │                  │  Creates neurons, regions,  │
└──────────────────────────────┘                  │  connectivity. Knows about  │
                                                  │  topology and rules.       │
                                                  │                              │
                                                  │  Experiment                 │
                                                  │  Orchestrates: what is      │
                                                  │  input, what is output,     │
                                                  │  how to feed, how to read.  │
                                                  └──────────────────────────────┘
```

**Why?** (supported by the literature)
- **Modular Deep Learning** (arXiv 2023): Separating computation from routing/organization
  enables autonomous modules, positive transfer, and systematic generalization.
- **PyTorch nn.Module**: The container is dumb, it only does `forward()`.
  It does not know if it is "input layer" or "output layer". That is decided by whoever composes.
- **Martin Fowler (Domain Model + Factory/Builder)**: Factory for creating lightweight
  elements (neurons), Builder for complex configurations (regions + connectivity).
- **Single Responsibility Principle**: Brain processes. Constructor organizes.
  Experiment orchestrates.

### 3.1 Class Diagram

```
═══════════════════════════════════════════════════════════════════
  STRUCTURE LAYER (core/) — Data containers, no processing logic
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────┐
│                       Brain                          │
│─────────────────────────────────────────────────────│
│  neuronas: list[Neurona]                            │
│  _neuronas_dict: dict[str, Neurona]                 │
│─────────────────────────────────────────────────────│
│  get_grid(w, h) → returns value matrix              │
│  get_neurona(id) → returns neuron by id             │
│─────────────────────────────────────────────────────│
│  Brain is a structure container — it holds neurons  │
│  but does NOT process them. Processing is done by   │
│  BrainTensor after compilation.                       │
│  Does NOT have regions.                             │
│  Does NOT know what is input or output.             │
└────────────┬────────────────────────────────────────┘
             │ contains N
             ▼
┌─────────────────────────────────────────────────────┐
│                     Neurona                         │
│─────────────────────────────────────────────────────│
│  id: str                                            │
│  valor: float {0, 1}                                │
│  dendritas: list[Dendrita]                          │
│  umbral: float                                      │
│─────────────────────────────────────────────────────│
│  activar_external(valor) → sets value from outside  │
│─────────────────────────────────────────────────────│
│  Data container only. The processing formulas       │
│  (fuzzy OR, tension, threshold) are implemented     │
│  in BrainTensor as vectorized tensor operations.      │
│                                                     │
│  ┌────────────────────────────────────────────┐     │
│  │         NeuronaEntrada (inherits)           │     │
│  │  No dendritas.                             │     │
│  │  Value is set externally.                  │     │
│  │  BrainTensor skips it during processing      │     │
│  │  via mascara_entrada.                      │     │
│  └────────────────────────────────────────────┘     │
└────────────┬────────────────────────────────────────┘
             │ contains M
             ▼
┌─────────────────────────────────────────────────────┐
│                    Dendrita                         │
│─────────────────────────────────────────────────────│
│  peso: float [-1, 1]    ← CAN BE NEGATIVE           │
│  sinapsis: list[Sinapsis]                           │
│─────────────────────────────────────────────────────│
│  Data container. The fuzzy AND formula              │
│  (avg(sinapsis) × peso) runs in BrainTensor.          │
│  Note: can have a SINGLE sinapsis if required.      │
└────────────┬────────────────────────────────────────┘
             │ contains K
             ▼
┌─────────────────────────────────────────────────────┐
│                    Sinapsis                          │
│─────────────────────────────────────────────────────│
│  peso: float [0, 1]     ← ALWAYS POSITIVE           │
│  neurona_entrante: Neurona (reference to axon)      │
│─────────────────────────────────────────────────────│
│  Data container. The recognition formula            │
│  (1 - |peso - entrada|) runs in BrainTensor.          │
└─────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════
  PROCESSING LAYER (core/) — Vectorized computation with PyTorch
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────┐
│                   BrainTensor                         │
│─────────────────────────────────────────────────────│
│  valores: Tensor          (neuron activations)      │
│  tensiones: Tensor        (surface tensions)        │
│  pesos_sinapsis: Tensor   (synapse weights)         │
│  indices_fuente: Tensor   (source neuron indices)   │
│  pesos_dendrita: Tensor   (dendrite weights)        │
│  mascara_valida: Tensor   (valid synapse mask)      │
│  mascara_entrada: Tensor  (input neuron mask)       │
│  umbrales: Tensor         (thresholds)              │
│─────────────────────────────────────────────────────│
│  procesar()               → one step (all neurons)  │
│  procesar_n(n)            → n steps in a row        │
│  get_grid(w, h)           → value matrix            │
│  get_tension_grid(w, h)   → tension matrix          │
│  get_valores()            → raw tensor              │
│  set_valor(idx, valor)    → modify one neuron       │
│─────────────────────────────────────────────────────│
│  This is the engine. All processing formulas run    │
│  here as vectorized tensor operations on GPU/CPU.   │
│  Compiled from Brain by ConstructorTensor.           │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                ConstructorTensor                    │
│─────────────────────────────────────────────────────│
│  compilar(brain, device) → BrainTensor      (static)  │
│─────────────────────────────────────────────────────│
│  Reads the OOP graph (Brain → Neurona → Dendrita → │
│  Sinapsis) and compiles it into flat tensors for    │
│  BrainTensor. This is the bridge between structure    │
│  and computation.                                   │
└─────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════
  ORGANIZATION LAYER (core/) — Does not know about processing
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────┐
│                    Region                           │
│─────────────────────────────────────────────────────│
│  nombre: str                                        │
│  neuronas: dict[str, Neurona]  ← references        │
│─────────────────────────────────────────────────────│
│  agregar(neurona)                                   │
│  ids() → list of ids                                │
│  valores() → list of values                         │
│  get_neurona(id) → returns neuron by id             │
│─────────────────────────────────────────────────────│
│  Does NOT own the neurons (reference only).         │
│  Brain and BrainTensor do not know regions exist.   │
│  It is a tool for Constructor and Experiment.       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                  Constructor                        │
│─────────────────────────────────────────────────────│
│  Creates neurons, groups them in regions,           │
│  builds connectivity (dendritas, sinapsis).         │
│─────────────────────────────────────────────────────│
│  key_by_coord(x, y) → neuron id         (static)   │
│  crear_grilla(w, h, filas_entrada,                  │
│    filas_salida, umbral) → Brain + regions           │
│  aplicar_mascara_2d(brain, w, h, mascara,           │
│    random_weights) → applies mask to all neurons    │
│  balancear_pesos(neuronas, target)                  │
│  balancear_sinapsis(neuronas, target)               │
│─────────────────────────────────────────────────────│
│  Knows about topology and connection patterns.      │
│  The ONLY one that knows how to wire the net.       │
│  Once built, ConstructorTensor compiles to tensors. │
└─────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════
  ORCHESTRATION LAYER (experiments/) — Uses everything above
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────┐
│              Experimento (base)                     │
│─────────────────────────────────────────────────────│
│  brain: Brain                                       │
│  regiones: dict[str, Region]                         │
│  width: int, height: int, generation: int           │
│─────────────────────────────────────────────────────│
│  setup(config)          → abstract                  │
│  step() → dict          → abstract                  │
│  step_n(count) → dict                               │
│  click(x, y)            → abstract                  │
│  reset()                → abstract                  │
│  get_frame()            → value grid                │
│  get_tension_frame()    → tension grid (or None)    │
│  get_stats() → dict                                 │
│  inspect(x, y) → dict                               │
│  is_complete() → bool                               │
└─────────────────────┬───────────────────────────────┘
                      │ inherits
                      ▼
┌─────────────────────────────────────────────────────┐
│            DeamonsLabExperiment                     │
│─────────────────────────────────────────────────────│
│  brain_tensor: BrainTensor                              │
│  _daemon_history: list                              │
│─────────────────────────────────────────────────────│
│  setup(config)    → Constructor + ConstructorTensor  │
│  step()           → brain_tensor.procesar()            │
│  reconnect(config)→ rewire without losing state     │
│  get_frame()      → brain_tensor.get_grid()            │
│  get_tension_frame() → brain_tensor.get_tension_grid() │
│  get_stats()      → daemon count, size, exclusion,  │
│                     stability, noise                │
│  click(x, y)      → toggle neuron value             │
│  reset()          → randomize + recompile           │
│─────────────────────────────────────────────────────│
│  The only experiment currently. Orchestrates the    │
│  full pipeline: build → compile → process → detect. │
└─────────────────────────────────────────────────────┘
```

### 3.2 Responsibility Flow

```
DeamonsLabExperiment (orchestrates)
  │
  │  1. setup: asks Constructor to build the network
  │
  ▼
Constructor (builds structure)
  │
  │  2. Creates neurons (Neurona and NeuronaEntrada)
  │  3. Groups them in regions
  │  4. Connects dendritas and sinapsis via aplicar_mascara_2d
  │  5. Delivers: Brain + dict of Regions
  │
  ▼
ConstructorTensor (compiles)
  │
  │  6. Reads the OOP graph (Brain → Neurona → Dendrita → Sinapsis)
  │  7. Flattens it into tensors (weights, indices, masks)
  │  8. Delivers: BrainTensor
  │
  ▼
BrainTensor (processes) — Brain and Regions remain as references
  │
  │  9. Experiment calls brain_tensor.procesar():
  │     - Synapse:  1 - |peso - source_value|   (vectorized)
  │     - Dendrite: avg(synapses) × peso         (vectorized)
  │     - Neuron:   max(exc) + min(inh) → tension
  │     - Threshold: tension > umbral → valor = 1, else 0
  │     - Input neurons skipped via mascara_entrada
  │
  ▼
Experiment reads brain_tensor.get_grid() → frame → WebSocket → Frontend
```

### 3.3 Processing Logic

All formulas run inside `BrainTensor.procesar()` as vectorized tensor
operations. The OOP classes (Neurona, Dendrita, Sinapsis) define the
structure; BrainTensor executes the math.

```
SYNAPSE:    value = 1 - |peso - source_neuron.valor|
            If peso=1 and source=1 → 1 (perfect match)
            If peso=0 and source=0 → 1 (perfect match)
            If peso=1 and source=0 → 0 (no match)
            If peso=0 and source=1 → 0 (no match)

DENDRITE:   value = average(synapse values) × peso_dendrita
            Fuzzy AND: all synapses must match
            peso_dendrita can be negative → inhibition

NEURON:     max_exc = max(dendrites where peso > 0)
            min_inh = min(dendrites where peso < 0)
            tension = max_exc + min_inh           (competition)
            If tension > umbral → valor = 1
            Else → valor = 0
            Fuzzy OR: any positive dendrite can activate
            But negative dendrites can inhibit

INPUT NEURON (NeuronaEntrada):
            Skipped by BrainTensor via mascara_entrada.
            Value is set externally via activar_external(valor).
```

### 3.4 Weight Rules

```
SINAPSIS.peso ∈ [0, 1]     ← Always positive or zero
                               Represents "pattern recognition"
                               peso≈1 recognizes entrada=1
                               peso≈0 recognizes entrada=0

DENDRITA.peso ∈ [-1, 1]    ← Can be negative
                               peso > 0: excitatory dendrita
                               peso < 0: inhibitory dendrita
                               Allows implementing NOT/inhibition
```

### 3.5 PyTorch Analogy

```
PyTorch                          NeuroFlow
─────────────────────────────    ─────────────────────────────
nn.Module (forward)          →   BrainTensor (procesar)
  Does not know if input/output    Does not know about regions
  Only computes                    Only processes tensors

Model definition (layers)    →   Brain + Constructor
  Defines topology                 Builds the OOP structure
  Knows about connectivity         Defines connectivity

torch.compile / JIT          →   ConstructorTensor (compilar)
  Optimizes the computation graph  Compiles OOP → flat tensors

Training loop                →   DeamonsLabExperiment
  Feeds data                       Feeds inputs
  Reads outputs                    Reads the grid
  Orchestrates everything          Orchestrates everything
```

### 3.6 Wiring Masks (masks.py)

Masks define each neuron's connection topology: which neighbors are
excitatory and which are inhibitory. They are configured as presets in `backend/core/masks.py`
and loaded dynamically from the API.

```
         Excitation (E)          Gap (G)           Inhibition (I)
        ┌───────────┐      ┌──────────────┐      ┌───────────────┐
        │  Moore r=n │      │  no connection │      │  ring/crown   │
        │  (close    │      │  (silence)     │      │  8 dendritas  │
        │  neighbors)│      │                │      │  sectorized   │
        └───────────┘      └──────────────┘      └───────────────┘
```

#### Wiring Nomenclature for Daemons


Daemon wiring masks use the convention `E G I [DE DI]`:

```
E<n>   Excitatory radius: Moore r=n
G<n>   Gap: n rings of silence between excitation and inhibition
I<n>   Inhibitory radius: n crown rings
DE<n>  Excitatory density: fraction 1/n of sinapsis (random, fixed seed)
DI<n>  Inhibitory density: fraction 1/n of sinapsis (random, fixed seed)
```

Example: `E2 G3 I3 DE1 DI1.5` → full Moore r=2, 3 gap rings,
3 inhibitory rings with ~67% density.

DE/DI omitted implies density 1 (full). Density uses `_random_sparse()`
with fixed seed so the mask is deterministic across runs but with
random spatial distribution (unlike `_sparse_ring` which uses
checkerboard-like patterns).

#### Generation Helpers

| Helper | Description |
|--------|-------------|
| `_moore(r)` | Moore neighborhood: Chebyshev dist ≤ r |
| `_ring(r_in, r_out)` | Ring: Chebyshev dist ∈ [r_in, r_out] |
| `_von_neumann(r)` | Von Neumann neighborhood: Manhattan dist ≤ r |
| `_sparse_ring(r_in, r_out, step)` | Deterministic sparse ring (checkerboard) |
| `_random_sparse(offsets, density, seed)` | Random subsampling with fixed seed |
| `_make_inhibitory(offsets, peso, n)` | Partitions offsets into n inhibitory sectors |
| `_partition(offsets, n)` | Divides offsets into n angular sectors |

---

## 4. Model Validation: Wolfram Automata

Wolfram elementary cellular automata (1D, 256 rules) were the first
test of the Synapse→Dendrite→Neuron system. By expressing arbitrary
logic rules as wiring patterns, they validated that the connectionist
model is **computationally expressive** — Rule 110 in particular is
Turing-complete.

Today, Wolfram rules are **wiring presets inside Deamons Lab** (`rule_110`,
`rule_30`), selectable from the same dropdown as Daemon masks. They
remain as a demonstration of compatibility: the same neural model that
produces daemons can also reproduce deterministic automata.

### 4.1 How It Works

The 2D grid shows the space-time diagram: each row is one generation
of the automaton. The bottom row is input (initial condition), and
activity propagates upward — one row per step — due to data
dependencies.

```
      Columns (space)
      ←─────────────────────────────→

  ↑   ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐   Row 0: last generation
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤   Each row reads from
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤   the row below it
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤   (flow ↑)
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  ↓   └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘   Bottom row: INPUT
                                        (single center cell active)
```

`BrainTensor.procesar()` processes all neurons in parallel, but since
each neuron reads from the row below and rows above the frontier
have zero inputs, the automaton naturally fills one row per step.

### 4.2 Neural Wiring for Rule 110 (Turing-complete)

Rule 110 in binary: `01101110`

| Pattern (left, center, right) | Decimal | Result |
|------------------------------|---------|--------|
| 1, 1, 1                      | 7       | **0**  |
| 1, 1, 0                      | 6       | **1**  |
| 1, 0, 1                      | 5       | **1**  |
| 1, 0, 0                      | 4       | **0**  |
| 0, 1, 1                      | 3       | **1**  |
| 0, 1, 0                      | 2       | **1**  |
| 0, 0, 1                      | 1       | **1**  |
| 0, 0, 0                      | 0       | **0**  |

Each neuron at position (x, y) connects to the 3 neurons in the row
below: (x-1, y+1), (x, y+1), (x+1, y+1). Threshold is set to 0.99.

```
Row y:     [  ?  ]  ← neuron to compute
              / | \
Row y+1: [left][cen][right]  ← 3 inputs
```

**Implementation with 5 dendrites** (one per pattern that produces 1):

```
Dendrite 1 → pattern 110: synapse weights [1, 1, 0] → peso_dendrita = +1
Dendrite 2 → pattern 101: synapse weights [1, 0, 1] → peso_dendrita = +1
Dendrite 3 → pattern 011: synapse weights [0, 1, 1] → peso_dendrita = +1
Dendrite 4 → pattern 010: synapse weights [0, 1, 0] → peso_dendrita = +1
Dendrite 5 → pattern 001: synapse weights [0, 0, 1] → peso_dendrita = +1
```

When the input pattern is, for example, `1 1 0`:
- Dendrite 1 (110): synapses → [1-|1-1|, 1-|1-1|, 1-|0-0|] = [1, 1, 1] → avg=1.0 ✓
- Dendrite 2 (101): synapses → [1-|1-1|, 1-|0-1|, 1-|1-0|] = [1, 0, 0] → avg=0.33 ✗
- ...only dendrite 1 gives high value → neuron activates → **1** ✓

The mask is generated automatically by `_wolfram_mask(rule)` in
`masks.py` — any Wolfram rule (0–255) can be added as a preset.

### 4.3 Implemented Rules

| Preset | Rule | Type | Description |
|--------|------|------|-------------|
| `rule_110` | Rule 110 | Turing-complete | Universally expressive |
| `rule_30` | Rule 30 | Chaotic | Pseudo-random, Sierpinski-like |

The neural model does not change between rules — only the dendrite
configuration differs. This was the first validation that the
Synapse→Dendrite→Neuron system can express arbitrary boolean logic.

---

## 5. API and Communication

### 5.1 REST Endpoints

```
GET  /api/experiments
     → [{ id: "deamons_lab", name: "Deamons Lab", masks: [...] }]

GET  /api/experiments/:id
     → { id, name, description, default_config: { width: 50, height: 50, mask: "deamon_3_en_50" } }

GET  /api/health
     → { status: "ok", version: "0.1.0" }
```

### 5.2 WebSocket Protocol

```
Connection: ws://host/ws/experiment

─── Client → Server ───────────────────────────────────

{ "action": "start",
  "experiment": "deamons_lab",
  "config": { "width": 30, "height": 30, "mask": "simple" } }

{ "action": "click", "x": 25, "y": 49 }    // Activate neuron

{ "action": "step" }                         // Advance 1 frame
{ "action": "play" }                         // Continuous animation
{ "action": "pause" }                        // Pause
{ "action": "reset" }                        // Restart

─── Server → Client ───────────────────────────────────

{ "type": "frame",
  "generation": 5,
  "grid": [[0,1,0,...], [1,1,0,...], ...],   // 50x50 matrix
  "stats": {
    "active_cells": 123,
    "processed_rows": 5,
    "total_rows": 50
  }
}

{ "type": "status",
  "state": "running" | "paused" | "ready" | "complete" }

{ "type": "error",
  "message": "..." }
```

### 5.3 Data Flow

```
┌──────────────────────┐          ┌──────────────────────────────────┐
│      FRONTEND        │          │            BACKEND               │
│                      │          │                                  │
│  ┌────────────────┐  │  start   │  ┌────────────────────────────┐  │
│  │   Sidebar      │──┼─────────►│  │   DeamonsLabExperiment     │  │
│  │  (wiring menu) │  │          │  │                            │  │
│  └────────────────┘  │          │  │  setup:                    │  │
│                      │          │  │   Constructor → Brain      │  │
│  ┌────────────────┐  │  click   │  │   ConstructorTensor →      │  │
│  │  PixelCanvas   │──┼─────────►│  │     BrainTensor              │  │
│  │  (HTML5 Canvas)│  │          │  │                            │  │
│  │  50×50 pixels  │  │          │  │  click(x,y):               │  │
│  └────────▲───────┘  │          │  │   brain_tensor.set_valor()   │  │
│           │          │          │  │   (toggle neuron)          │  │
│  ┌────────┴───────┐  │  frame   │  │                            │  │
│  │  useExperiment │◄─┼──────────│  │  step:                     │  │
│  │  (WebSocket)   │  │          │  │   brain_tensor.procesar()    │  │
│  └────────────────┘  │          │  │   brain_tensor.get_grid()    │  │
│                      │          │  │     → frame                │  │
│  ┌────────────────┐  │          │  │                            │  │
│  │  Controls      │──┼─────────►│  │  reconnect:               │  │
│  │  Play/Pause    │  │  step    │  │   rewire without losing    │  │
│  └────────────────┘  │          │  │   neuron state             │  │
│                      │          │  └────────────────────────────┘  │
└──────────────────────┘          └──────────────────────────────────┘
```

---

---

## 7. Free Hosting

### 7.1 Backend → Render.com

- **Plan**: Free tier (750 hours/month)
- **Limitation**: Spin-down after 15min of inactivity (~1min cold start)
- **Deploy**: From Git, auto-build with `requirements.txt`
- **Runtime**: Python 3.11, uvicorn

```yaml
# render.yaml (Blueprint)
services:
  - type: web
    name: neuroflow-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

### 7.2 Frontend → Vercel

- **Plan**: Hobby (free)
- **Build**: Vite produces static files
- **Deploy**: From Git, auto-detect Vite

```json
// vercel.json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite"
}
```

### 7.3 CORS

The backend must allow requests from the frontend:

```python
# main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://neuroflow.vercel.app", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 8. Roadmap

See **[Stages](STAGES.md)** for the full project roadmap (5 stages:
Daemons → SOM → Motor/Nociceptor → Tuning → Motor Agents).

---

## 9. Architecture Decision Records

Key technical decisions are documented as individual ADR files following
the [ADR pattern](https://adr.github.io/) (Michael Nygard, 2011).

See **[decisions/](decisions/)** for the full list, including:

- Why a web app instead of Jupyter Notebooks
- Why separate frontend and backend
- Why WebSocket instead of HTTP polling
- Why React over Svelte/Vue
- Why PyTorch for computation

---

← Back to [README](../README.md)

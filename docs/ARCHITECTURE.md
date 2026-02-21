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
│   NumPy for matrix operations                           │
│   Deploy: Render.com (free, 750h/month)                  │
└─────────────────────────────────────────────────────────┘
```

### Rationale

| Component | Choice | Why |
|-----------|--------|-----|
| Backend framework | **FastAPI** | Native async, WebSocket, typing, most popular in Python 2025-2026 |
| Backend runtime | **uvicorn** | Standard ASGI server for FastAPI |
| Compute | **NumPy** | Vectorized matrix operations, releases GIL |
| Frontend bundler | **Vite** | Instant build, HMR, current standard |
| Frontend framework | **React 19 + TypeScript** | Most adopted, reusable components |
| Rendering | **HTML5 Canvas** | Direct, fast, perfect for pixel grids |
| Communication | **WebSocket** | Bidirectional, low latency, ideal for frame streaming |
| Backend hosting | **Render.com** | Only one with real free tier for Python (750h/month) |
| Frontend hosting | **Vercel** | Generous free tier, auto-deploy from Git, optimal for React |
| Backend tests | **pytest** | Python standard, simple, powerful |
| Frontend tests | **Vitest** | Native to Vite, Jest API compatible |

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
│   │   ├── red.py                 # Neural network (dumb container)
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
│       ├── test_red.py             # Red does NOT know about regions
│       ├── test_region.py          # Region is just grouping
│       ├── test_constructor.py     # Constructor assembles Red + Regions
│       ├── test_deamons_lab.py
│       └── test_red_tensor.py
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
│   └── AUTHOR.md                  # About the author and dedication
│
├── README.md                      # Entry point, navigation
└── .gitignore
```

---

## 3. Neural Model (Core)

Port of the RedJavaScript model to Python, with a key architectural improvement:
**separation of responsibilities between processing and organization**.

### 3.0 Design Principle: Separation of Responsibilities

In the original project (RedJavaScript), the `Red` class knew about regions
(INPUT, OUTPUT, INTERNAL) and decided which neurons to process. This couples
organization with processing.

In NeuroFlow we separate these responsibilities:

```
PROCESSING (does not know about organization)     ORGANIZATION (does not know about processing)
┌──────────────────────────────┐                  ┌──────────────────────────────┐
│  Red                         │                  │  Region                      │
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
- **Single Responsibility Principle**: Red processes. Constructor organizes.
  Experiment orchestrates.

### 3.1 Class Diagram

```
═══════════════════════════════════════════════════════════════════
  PROCESSING LAYER (core/) — Does not know about organization
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────┐
│                        Red                          │
│─────────────────────────────────────────────────────│
│  neuronas: dict[str, Neurona]                       │
│─────────────────────────────────────────────────────│
│  procesar()     → processes ALL neurons             │
│  get_grid(w, h) → returns value matrix              │
│  get_neurona(id) → returns neuron by id             │
│─────────────────────────────────────────────────────│
│  Does NOT have regions.                             │
│  Does NOT know what is input or output.             │
│  Only iterates and processes what it was given.     │
└────────────┬────────────────────────────────────────┘
             │ contains N
             ▼
┌─────────────────────────────────────────────────────┐
│                     Neurona                         │
│─────────────────────────────────────────────────────│
│  id: str                                            │
│  valor: float {0, 1}                                │
│  tension_superficial: float [-1, 1]                 │
│  dendritas: list[Dendrita]                          │
│  umbral: float                                      │
│─────────────────────────────────────────────────────│
│  procesar()   → fuzzy OR of dendritas               │
│  activar()    → threshold over tension              │
│  entrenar()   → propagates training                 │
│─────────────────────────────────────────────────────│
│                                                     │
│  ┌────────────────────────────────────────────┐     │
│  │         NeuronaEntrada (inherits)           │     │
│  │  No dendritas.                             │     │
│  │  procesar() → no-op                        │     │
│  │  activar()  → no-op                        │     │
│  │  activar_external(valor) → sets value      │     │
│  │                                            │     │
│  │  Red processes it the same as others,      │     │
│  │  but internally it does nothing.           │     │
│  │  Red does NOT need to know it is special.  │     │
│  └────────────────────────────────────────────┘     │
└────────────┬────────────────────────────────────────┘
             │ contains M
             ▼
┌─────────────────────────────────────────────────────┐
│                    Dendrita                         │
│─────────────────────────────────────────────────────│
│  peso: float [-1, 1]    ← CAN BE NEGATIVE           │
│  valor: float                                       │
│  sinapsis: list[Sinapsis]                           │
│─────────────────────────────────────────────────────│
│  procesar()   → avg(sinapsis) * peso  (fuzzy AND)   │
│  entrenar()   → propagates to sinapsis              │
│─────────────────────────────────────────────────────│
│  Note: can have a SINGLE sinapsis if required       │
└────────────┬────────────────────────────────────────┘
             │ contains K
             ▼
┌─────────────────────────────────────────────────────┐
│                    Sinapsis                          │
│─────────────────────────────────────────────────────│
│  peso: float [0, 1]     ← ALWAYS POSITIVE           │
│  valor: float                                       │
│  neurona_entrante: Neurona (reference to axon)      │
│─────────────────────────────────────────────────────│
│  procesar()   → 1 - |peso - entrada|                │
│  entrenar()   → Hebbian: peso += (entrada - peso)*η │
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
│─────────────────────────────────────────────────────│
│  Does NOT own the neurons (reference only).         │
│  Red does not know regions exist.                   │
│  It is a tool for Constructor and Experiment,      │
│  not for Red.                                       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                  Constructor                        │
│─────────────────────────────────────────────────────│
│  Creates neurons, groups them in regions,           │
│  builds connectivity (dendritas, sinapsis).         │
│─────────────────────────────────────────────────────│
│  crear_grilla(w, h)  → Red + dict of regions        │
│  crear_region(nombre, neuronas) → Region            │
│  conectar(origen, destino, mascara_relativa)        │
│  aplicar_regla_wolfram(regla, neuronas, vecinos)    │
│─────────────────────────────────────────────────────│
│  Knows about topology and connection patterns.      │
│  It is the ONLY one that knows how to wire the net. │
│  Once built, Red works on its own.                  │
└─────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════
  ORCHESTRATION LAYER (experiments/) — Uses everything above
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────┐
│              Experiment (base)                      │
│─────────────────────────────────────────────────────│
│  red: Red                                           │
│  regiones: dict[str, Region]                         │
│─────────────────────────────────────────────────────│
│  setup(config)  → uses Constructor to build all     │
│  step()         → red.procesar() + returns frame    │
│  click(x, y)    → finds neuron in input region       │
│  reset()        → restarts                           │
│  get_frame()    → red.get_grid()                     │
│─────────────────────────────────────────────────────│
│  KNOWS which region is input and which is output.   │
│  KNOWS how to feed the network and read results.    │
│  Red knows nothing about this.                      │
└─────────────────────────────────────────────────────┘
```

### 3.2 Responsibility Flow

```
Experiment (orchestrates)
  │
  │  1. setup: asks Constructor to build the network
  │
  ▼
Constructor (organizes)
  │
  │  2. Creates neurons (Neurona and NeuronaEntrada)
  │  3. Groups them in regions
  │  4. Connects dendritas and sinapsis according to the rule
  │  5. Delivers: Red + dict of Regions
  │
  ▼
Red (processes) ◄── Regions (references)
  │
  │  6. Experiment calls red.procesar()
  │  7. Red iterates ALL neurons:
  │     - NeuronaEntrada.procesar() → no-op (it already has its value)
  │     - Neurona.procesar() → evaluates dendritas → sinapsis
  │  8. Red iterates ALL neurons:
  │     - NeuronaEntrada.activar() → no-op
  │     - Neurona.activar() → threshold over tension
  │
  ▼
Experiment reads red.get_grid() → frame → WebSocket → Frontend
```

### 3.3 Processing Logic

```
SINAPSIS:   valor = 1 - |peso - neurona_entrante.valor|
            If peso=1 and entrada=1 → 1 (perfect match)
            If peso=0 and entrada=0 → 1 (perfect match)
            If peso=1 and entrada=0 → 0 (no match)
            If peso=0 and entrada=1 → 0 (no match)

DENDRITA:   valor = average(sinapsis.procesar()) × peso_dendrita
            Fuzzy AND: all sinapsis must match
            peso_dendrita can be negative → inhibition

NEURONA:    max_dendrita = max(dendritas.valor)
            min_dendrita = min(dendritas.valor)   (negatives)
            tension = max + min                   (competition)
            If tension > umbral → valor = 1
            Else → valor = 0
            Fuzzy OR: any positive dendrita can activate
            But negative dendritas can inhibit

NEURONA_ENTRADA:
            procesar() → no-op (no dendritas)
            activar()  → no-op (its value was already set)
            Only changes via activar_external(valor) from Experiment
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
nn.Module (forward)          →   Red (procesar)
  Does not know if input/output    Does not know about regions
  Only computes                    Only iterates neurons

nn.Sequential / Model        →   Constructor
  Composes modules in order        Builds Red with regions
  Defines topology                 Defines connectivity

Training loop                →   Experiment
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

#### Daemon Nomenclature

Daemon-type masks use the convention `E G I [DE DI]`:

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

## 4. Experiment 0: Elementary Automaton (Von Neumann)

### 4.1 Concept

An elementary cellular automaton (1D, Wolfram rules) implemented
entirely with the neural model. The 2D grid shows the space-time
diagram: each row is one generation of the automaton.

```
      Columns (space, 50 cells)
      ←─────────────────────────────→

  ↑   ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐   Row 0: OUTPUT (last generation)
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤   Internal rows: INTERNAL
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤   (processed bottom-up)
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
Flow ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │   └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘   Row 49: INPUT (initial condition)
                                        ← User clicks here
```

### 4.2 Neural Wiring for Rule 111

Rule 111 in binary: `01101111`

| Pattern (left, center, right) | Decimal | Result |
|------------------------------|---------|--------|
| 1, 1, 1                      | 7       | **0**  |
| 1, 1, 0                      | 6       | **1**  |
| 1, 0, 1                      | 5       | **1**  |
| 1, 0, 0                      | 4       | **0**  |
| 0, 1, 1                      | 3       | **1**  |
| 0, 1, 0                      | 2       | **1**  |
| 0, 0, 1                      | 1       | **1**  |
| 0, 0, 0                      | 0       | **1**  |

Each internal neuron at position (x, y) connects to the 3 neurons
in the row below: (x-1, y+1), (x, y+1), (x+1, y+1).

```
Row y:     [  ?  ]  ← neuron to compute
              / | \
Row y+1: [left][cen][right]  ← 3 inputs
```

**Implementation with 6 dendritas** (one per pattern that produces 1):

```
Dendrita 1 → pattern 110: sinapsis weights [1, 1, 0] → peso_dendrita = +1
Dendrita 2 → pattern 101: sinapsis weights [1, 0, 1] → peso_dendrita = +1
Dendrita 3 → pattern 011: sinapsis weights [0, 1, 1] → peso_dendrita = +1
Dendrita 4 → pattern 010: sinapsis weights [0, 1, 0] → peso_dendrita = +1
Dendrita 5 → pattern 001: sinapsis weights [0, 0, 1] → peso_dendrita = +1
Dendrita 6 → pattern 000: sinapsis weights [0, 0, 0] → peso_dendrita = +1
```

When the input pattern is, for example, `1 1 0`:
- Dendrita 1 (110): sinapsis → [1-|1-1|, 1-|1-1|, 1-|0-0|] = [1, 1, 1] → avg=1.0 ✓
- Dendrita 2 (101): sinapsis → [1-|1-1|, 1-|0-1|, 1-|1-0|] = [1, 0, 0] → avg=0.33 ✗
- ...only dendrita 1 gives high value → neuron activates → **1** ✓

### 4.3 Frame-by-Frame Processing

```
Frame 0:  Only row 49 visible (INPUT, user click)
Frame 1:  Row 48 processed (reads row 49)
Frame 2:  Row 47 processed (reads row 48)
...
Frame 49: Row 0 processed (OUTPUT)

Total: 49 frames to fill the entire grid
```

### 4.4 Additional Rules Planned

| Rule | Type | Description |
|------|------|-------------|
| Rule 111 | Deterministic | First test, many 1s |
| Rule 30 | Chaotic | Sierpinski triangles, chaos |
| Rule 90 | Fractal | Perfect Sierpinski triangle |
| Rule 110 | Turing-complete | Theoretically most interesting |

Each rule only requires reconfiguring which dendritas each neuron has.
The neural model (Sinapsis, Dendrita, Neurona, Red) does not change.

---

## 5. API and Communication

### 5.1 REST Endpoints

```
GET  /api/experiments
     → [{ id: "deamons_lab", name: "Deamons Lab", masks: [...] }]

GET  /api/experiments/:id
     → { id, name, description, default_config: { width: 30, height: 30, mask: "simple" } }

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
│  │   Sidebar      │──┼─────────►│  │   Experiment (orchestrates)│  │
│  │  (experiments) │  │          │  │                            │  │
│  └────────────────┘  │          │  │  setup:                    │  │
│                      │          │  │   Constructor → Red        │  │
│  ┌────────────────┐  │  click   │  │               + Regions     │  │
│  │  PixelCanvas   │──┼─────────►│  │                            │  │
│  │  (HTML5 Canvas)│  │          │  │  click(x,y):               │  │
│  │  50×50 pixels  │  │          │  │   region_entrada           │  │
│  └────────▲───────┘  │          │  │     .get(x,y)              │  │
│           │          │          │  │     .activar_external(1)   │  │
│  ┌────────┴───────┐  │  frame   │  │                            │  │
│  │  useExperiment │◄─┼──────────│  │  step:                     │  │
│  │  (WebSocket)   │  │          │  │   red.procesar()  ← dumb   │  │
│  └────────────────┘  │          │  │     Neurona.procesar()     │  │
│                      │          │  │       Dendrita.procesar()  │  │
│  ┌────────────────┐  │          │  │         Sinapsis.procesar()│  │
│  │  Controls      │──┼─────────►│  │   red.get_grid() → frame  │  │
│  │  Play/Pause    │  │  step    │  │                            │  │
│  └────────────────┘  │          │  └────────────────────────────┘  │
└──────────────────────┘          └──────────────────────────────────┘
```

---

## 6. Frontend: UI Design

```
┌─────────────────────────────────────────────────────────────┐
│  NeuroFlow                                          v0.1.0  │
├──────────────┬──────────────────────────────────────────────┤
│              │                                              │
│ EXPERIMENTS  │         NEURON GRID                          │
│              │                                              │
│ ● Exp 0:    │    ┌──────────────────────────────┐           │
│   Von Neumann│    │  ■ □ □ ■ □ ■ ■ □ □ ■ ...  │  OUTPUT    │
│              │    │  □ ■ □ □ ■ □ □ ■ □ □ ...  │           │
│   Rule:      │    │  ■ ■ □ ■ ■ ■ □ □ ■ □ ...  │           │
│   [111 ▼]    │    │  □ □ ■ □ □ □ ■ □ □ ■ ...  │           │
│              │    │  ...                        │           │
│   Size:      │    │  □ □ □ □ □ ■ □ □ □ □ ...  │  INTERNAL  │
│   50 × 50    │    │  □ □ □ □ □ □ □ □ □ □ ...  │           │
│              │    └──────────────────────────────┘           │
│   Speed:     │    ← Click to activate neurons →             │
│   ████░░ 7fps│                                              │
│              │   ┌──────────────────────────────────┐       │
│ ○ Exp 1:    │   │  ▶ Play  ⏸ Pause  ⏭ Step  ↺ Reset │    │
│   Conway     │   └──────────────────────────────────┘       │
│   (next)     │                                              │
│              │   Gen: 23/50  │  Active cells: 147           │
│              │                                              │
├──────────────┴──────────────────────────────────────────────┤
│  ⬛ = active neuron (valor=1)   □ = inactive (valor=0)      │
│  🔵 = INPUT   🔴 = OUTPUT   ⬜ = INTERNAL                   │
└─────────────────────────────────────────────────────────────┘
```

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

## 8. Implementation Plan

### Phase 0: Walking Skeleton (this sprint)

```
1. [Tests]  → test_sinapsis.py, test_dendrita.py, test_neurona.py, test_red.py
2. [Core]   → sinapsis.py, dendrita.py, neurona.py, red.py, constructor.py
3. [Tests]  → test_deamons_lab.py, test_red_tensor.py
4. [API]    → main.py with WebSocket + experiments endpoint
5. [UI]     → React app with Canvas, sidebar, controls
6. [Exp]    → Deamons Lab (all wirings + Wolfram) end-to-end
7. [Deploy] → Backend on Render, Frontend on Vercel
```

### Phase 1: More Rules + Conway

```
8.  Rule 30, 90, 110 (dendrita reconfiguration only)
9.  Experiment 1: Game of Life (Conway) - Moore neighborhood (8 neighbors)
10. UI: dynamic experiment selector
```

### Phase 2: Emergent Learning

```
11. Activate Hebbian training
12. Synaptic pruning
13. Real-time weight visualization
```

### Phase 3: Daemons + HTM

```
14. Self-organized maps
15. Hierarchical temporal memory
16. Functional regions (PAIN)
```

---

## 9. Key Technical Decisions

### Why not Jupyter Notebooks?

- Not deployable as a web application
- Require local installation
- Interactive visualization is limited
- Do not scale to multiple users

### Why separate frontend and backend?

- Neural computation can be heavy → dedicated backend
- UI must be responsive → do not block with computation
- Allows independent scaling
- Allows using GPU in backend without affecting UI

### Why WebSocket and not polling?

- The automaton produces ~10-30 frames/second
- Polling would generate too many HTTP requests
- WebSocket enables continuous bidirectional streaming
- Client can send clicks without extra latency

### Why React and not Svelte/Vue?

- React is the most adopted and documented
- For a Canvas with sidebar, React is sufficiently simple
- Larger ecosystem for future needs
- Mature TypeScript support

### Why NumPy for computation?

- Vectorized operations are ~100x faster than Python loops
- For 50×50 = 2500 neurons, it is instant
- Scales well up to ~1000×1000 without GPU
- Familiar to scientists and engineers

---

← Back to [README](../README.md)

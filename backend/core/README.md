# Neural Model — `backend/core/`

The heart of NeuroFlow. This is where the connectionist model lives: the
fundamental pieces that process information without a central controller.

---

## In one line

> Synapses recognize patterns → Dendrites combine them (excite or inhibit)
> → Neurons compete and activate → the Network processes them all without
> knowing anything about organization.

---

## Fundamental pieces

```
Synapse (weight ≥ 0)  →  Dendrite (weight ∈ [-1,1])  →  Neuron (active/inactive)
   recognizes pattern       fuzzy AND + inhibition          competitive fuzzy OR
```

### Synapse (`sinapsis.py`)

The minimal unit of connection. Connects an incoming neuron to a dendrite.

- **Weight**: always in `[0, 1]` — represents pattern recognition
- **Processing**: `value = 1 - |weight - input|`
  - `weight≈1` recognizes `input=1` (perfect match → 1)
  - `weight≈0` recognizes `input=0` (perfect match → 1)
  - Mismatch → low value
- **Training**: Hebbian — `weight += (input - weight) × η`

### Dendrite (`dendrita.py`)

Input branch that groups synapses and combines them.

- **Weight**: in `[-1, 1]` — can be negative (inhibition)
- **Processing**: `value = average(synapses) × dendrite_weight` (fuzzy AND)
- Positive weight → **excitatory** dendrite
- Negative weight → **inhibitory** dendrite
- Can have a single synapse if needed

### Neuron (`neurona.py`)

The decision unit. Contains dendrites and resolves the competition.

- **Value**: `{0, 1}` — active or inactive
- **Surface tension**: `[-1, 1]` — accumulation before threshold
- **Processing**:
  1. `max_dendrite = max(dendrites.value)` (excitation)
  2. `min_dendrite = min(dendrites.value)` (inhibition, if negative)
  3. `tension = max + min` (competition)
  4. If `tension > threshold` → active (1), otherwise → inactive (0)
- Fuzzy OR: any positive dendrite can activate
- But negative dendrites can inhibit

**InputNeuron** (inherits from Neuron): no dendrites, its value is set
externally. The Network processes it the same way but it does nothing internally.

### Network (`red.py`)

"Dumb" container — knows nothing about organization, only processes.

- Contains a dictionary of neurons
- `procesar()`: iterates all and processes them (includes InputNeuron, which is a no-op)
- `get_grid(w, h)`: returns the value matrix for visualization
- **Does not know** what input, output, region, or experiment is

---

## Organization (without processing)

### Region (`region.py`)

Named grouping of neurons — references only, not owner.
The Network does not know that regions exist.

### Constructor (`constructor.py`)

Factory/Builder that assembles the network: creates neurons, groups them
into regions, builds connectivity (dendrites, synapses) according to rules.

**It is the only one that knows how to wire the network.** Once built,
the Network runs on its own.

### ConstructorTensor (`constructor_tensor.py`)

Tensorized version of the constructor for efficient matrix computation.

### NetworkTensor (`red_tensor.py`)

Tensorized version of the Network that operates with matrices instead of
individual Python objects.

---

## Wiring masks (`masks.py`)

Masks define the connection topology: which neighbors are excitatory, which
are inhibitory, and with what density.

### Daemon nomenclature: `E G I [DE DI]`

| Parameter | Meaning | Example |
|-----------|---------|---------|
| **E***n* | Excitatory radius (Moore r=*n*) | E3 = Moore r=3 (48 neighbors) |
| **G***n* | Gap: silence rings | G12 = 12 rings without connection |
| **I***n* | Inhibitory radius (corona) | I3 = 3 inhibitory rings |
| **DE***n* | Excitatory density (1/*n*) | DE1 = full, DE3 = ~33% |
| **DI***n* | Inhibitory density (1/*n*) | DI1 = full, DI1.5 = ~67% |

```
      Excitation (E)         Gap (G)          Inhibition (I)
    ┌─────────────┐    ┌──────────────┐    ┌───────────────┐
    │  Moore r=n  │    │  no connection│    │  ring/corona  │
    │  (close     │    │  (silence)    │    │  sectorized   │
    │  neighbors) │    │              │    │  (8 dendrites)  │
    └─────────────┘    └──────────────┘    └───────────────┘
```

**Example**: `E2 G3 I3 DE1 DI1.5`
- Excitation: Moore r=2 full (24 neighbors)
- Gap: r=3–5 (3 silence rings)
- Inhibition: r=6–8 with density 1/1.5 ≈ 67%

### Biological inspiration

This structure replicates the **Mexican hat** observed in the visual cortex
by Hubel & Wiesel (1962) and formalized by Kohonen in SOMs: local excitation
surrounded by lateral inhibition — the fundamental pattern that enables
topographic self-organization in the brain.

---

## Design principle: separation of responsibilities

```
PROCESSING                            ORGANIZATION
(knows nothing about organization)    (knows nothing about processing)

  Network                               Region
  Only processes neurons.               Only groups neurons.
  Knows nothing about regions.          Does not process them.

                                        Constructor
                                        Creates and wires.

                                        Experiment
                                        Orchestrates everything.
```

See [Technical Architecture](../../docs/ARCHITECTURE.md) for the complete
design with diagrams.

---

← Back to [README](../../README.md)

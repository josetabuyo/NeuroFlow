# Roadmap

The stages follow a logical order: each one builds on the previous one.
The ultimate goal is an intelligent motor agent that operates without language,
emulating simple creatures such as *Aplysia* or the zebrafish.

---

## Summary

```
Stage 1   Stage 2       Stage 3   Stage 4      Stage 5
Daemon  → Dynamic SOM → ITON    → Tuning     → Motor Agents
(✓)       (✓)           (next)    (genetic)    (simulation)
```

**ITON** = Input · Tissue · Output · Nociceptor

---

## Stage 1: Finding the Daemon ✓

**Status:** Essentially complete.

**Objective:** Discover whether a purely connectionist network, with local rules
and no centralized processing, can produce stable units of activation — the *daemons*.

**What was achieved:**

| Achievement | Description |
|-------------|-------------|
| Movement | Daemons move: up, down, left, right |
| Stability | They do not dissipate; they maintain their shape |
| Noise resistance | The signal prevails over noise |
| Competitive exclusion | Daemons compete and mutually exclude one another |
| Natural balance | ~50% of neurons active at any given time |
| Activation bubbles | When one turns off, another turns on; dynamic equilibrium |
| Convergence | When manipulated externally, the system converges to a new state |
| Multiple resolutions | Different connectomes store information at different scales |

**Experiment:** *Deamons Lab* — connectome laboratory with
presets such as `E G I DE DI` (see [Neuronal Model](../backend/core/README.md)).

**Unexpected finding:** Daemons behave like musical notes
with respect to competitive exclusion (see [Vision](VISION.md#paralelo-con-las-notas-musicales)).

**Next:** Will be revisited in Stage 4 (Tuning) with genetic algorithms.

---

## Stage 2: Dynamic SOM ✓

**Status:** Complete (May 2026).

**Objective:** Implement a Self-Organizing Map (SOM)
using NeuroFlow's connectionist model and observe whether the system
replicates the topographic organization capacity that Kohonen described.

**What we seek:**

- To see how the system behaves similarly to a classical SOM
- To improve how images are stored and organized
- To observe SOM dynamics for both:
  - Stable connectomes (fixed daemons)
  - Connectomes with movement (without losing clustering capability)
- To implement **training** within this stage

**Experiment:** *Dynamic SOM* — new experiment in the sidebar.

**Inspiration:** Kohonen (1990), Hubel & Wiesel (1981), the relationship
between SOMs and convolutional network layers (Deep Dream).

### Infrastructure built (April 2026)

| Feature | Description |
|---------|-------------|
| Unified `Experiment` class | Single class replaces per-experiment files. All features opt-in via config sections. |
| Synthetic input patterns | `HALF_TOP`, `HALF_BOT`, `BARS_H`, `BARS_V`, `DOT_TL`, `DOT_BR` — no font needed |
| Polynomial tension shaping | `tension_function: {"x": N, "x_pow_2": N, "x_pow_3": N}` — composable, soft param |
| **Soft activation** | `"activation": "soft"` on any wiring region → `value = clamp(tension, 0, 1)` instead of binary threshold; enables analog output for output/nociceptor regions |
| Config templates + history | Dropdown with presets, SQLite history per preset, JSON editor in UI |
| Per-dendrite-type learning | `lr_exc`, `lr_inh`, `lr_input` multipliers on top of `rate` — allows freezing recurrent weights while training only input dendrites |
| Input density | `input.density` (0–1) — fraction of input neurons each tissue neuron connects to; sparse connectivity promotes specialization |
| Auto-fit glyph rendering | Characters fill the input grid to the edge (`padding=0`); `padding=N` adds margin |
| WebSocket error handling | Backend exceptions now sent to client instead of silently closing the connection |
| Compile time optimization | `ConstructorTensor.compilar` went from 31s → 4s (bulk numpy array fill instead of per-element tensor writes) |

### Current experiment config (Dynamic SOM)

```json
{
  "grid": { "width": 50, "height": 50 },
  "wiring": {
    "mask": "deamon_e3_g2_i12_de1_di1",
    "dendrite_exc_weight": 0.9,
    "dendrite_inh_weight": -1,
    "process_mode": "avg_vs_avg",
    "tension_function": { "x": 3, "x_pow_3": 9, "x_pow_2": 2 }
  },
  "input": {
    "text": "HALF_TOP,HALF_BOT",
    "resolution": 20,
    "frames_per_char": 10,
    "dendrite_input_weight": 0.2
  },
  "learning": {
    "rate": 1.0,
    "lr_exc": 0.0,
    "lr_inh": 0.0,
    "lr_input": 0.01
  }
}
```

Only input dendrites learn (`lr_exc=0, lr_inh=0`). Recurrent (daemon) weights stay fixed.
This isolates whether the input pathway alone can produce topographic organization.

### Results observed (April 2026)

**The experiment confirmed topographic self-organization consistent with SOM behavior.**

| Observation | Description |
|-------------|-------------|
| Center-of-mass shift | Daemon clusters visibly migrated in response to each input pattern |
| Abrupt separation (orthogonal patterns) | With `HALF_TOP` / `HALF_BOT` (zero shared pixels), daemon populations for each pattern separated violently and cleanly |
| Graded separation (similar patterns) | With patterns sharing pixels, daemons that settled were clearly biased toward each pattern — proportionally, not abruptly — and this was consistent across runs |
| Input-only learning suffices | The recurrent lateral mask (excitatory center + inhibitory surround) does **not** need training. Only the input dendrite weights need to be learned to produce full SOM-like topographic organization |

### Key finding

The lateral connectivity structure (the daemon connectome) already implements the
competitive/cooperative dynamics that a classical Kohonen SOM achieves via an explicit
neighborhood function. The excitatory-center / inhibitory-surround mask provides:

- **Local cooperation** (neighboring neurons activate together → daemon cohesion)
- **Global competition** (distant neurons suppress each other → daemon exclusion)

This means the recurrent weights are effectively a *fixed topographic prior*, and
learning only needs to map input patterns onto that prior. This is a stronger result
than a standard SOM: the spatial structure is not learned — it emerges from the
connectome and is stable before any input is presented.

**Implication for Stage 3:** The motor/nociceptor system can be built on top of a
frozen daemon layer. Only the interface weights (input → daemon, daemon → output)
need to be trained.

---

## Stage 3: ITON — Input · Tissue · Output · Nociceptor

**Status:** Next (May 2026).

**Objective:** Add output (motor) and error-signal (nociceptor) regions on top of the
frozen daemon tissue. Build a minimal learning loop that does not require backpropagation:
the nociceptor provides a local error signal; the tissue routes it as inhibitory feedback.

---

### Architecture

```
┌──────────┐    connection     ┌──────────┐    connection    ┌──────────┐
│  input   │ ─────────────→   │  tissue  │ ──────────────→  │  output  │
│ (source) │  weight > 0       │ (daemon) │  weight > 0      │  (read)  │
└──────────┘                  └──────────┘                   └──────────┘
                                    ↑                              │
                              weight < 0                     compare vs
                                    │                         target
                              ┌─────────────┐                    │
                              │ nociceptor  │ ←──────────────────┘
                              │  (error)    │   diff(target, output)
                              └─────────────┘
```

| Region | Type | Lateral wiring | Learns |
|--------|------|---------------|--------|
| **input** | `NeuronaEntrada` — driven by source | none | no |
| **tissue** | daemon neurons — local exc/inh connectome | frozen daemon mask | no (prior from Stage 2) |
| **output** | regular neurons — no local wiring | none (Level 1), daemon mask (Level 2) | yes — `tissue→output` connection weights |
| **nociceptor** | `NeuronaEntrada` — driven by error signal | none | no |

**Connection flow in the canonical schema:**

```json
{
  "regions": [
    { "id": "input",       "grid": {...}, "source": {...} },
    { "id": "tissue",      "grid": {...}, "wiring": { "deamon": {...} } },
    { "id": "output",      "grid": {...} },
    { "id": "nociceptor",  "grid": {...}, "source": { "type": "error_diff" } }
  ],
  "connections": [
    { "from": "input",      "to": "tissue",  "weight":  0.35, "learning": { "rate": 0.0 } },
    { "from": "tissue",     "to": "output",  "weight":  1.0,  "learning": { "rate": 0.01 } },
    { "from": "nociceptor", "to": "tissue",  "weight": -0.5,  "learning": { "rate": 0.0 } }
  ]
}
```

---

### What already works (inherited from Stage 2)

| Feature | Where |
|---------|-------|
| Canonical `regions[]` + `connections[]` schema | `experiment.py` — `_to_canonical()` |
| Multiple regions with independent grids | `experiment.py` — `setup()` |
| Directed connections with per-type learning | `brain_tensor.py` — `es_input_syn`, `learn()` |
| `conn_exclude_range` dead zone on connection dendrites | `brain_tensor.py`, `experiment.py` |
| Per-neuron centroid jitter on local wiring | `constructor.py` — `centroid_jitter` |
| JSON editor with smart-quote and newline sanitization | `JsonConfigEditor.tsx` |

---

### What needs to be built

#### Backend

1. **Output region** — regular neurons with no local wiring, fed from tissue via a
   learned connection. The `Experiment` class must instantiate these and include them
   in the `BrainTensor` compilation alongside the tissue + input neurons.

2. **Label source** — a new `source.type: "label"` (for Experiment 1) that injects
   the ground-truth class signal directly into the output region as `NeuronaEntrada`.
   Knows which class is currently shown from the input source frame counter.

3. **Error-diff source** — a new `source.type: "error_diff"` (for Experiment 2) that
   computes `diff(target_frame, output_activation)` each step and injects the result
   into a nociceptor region.

4. **Multi-region BrainTensor** — extend `BrainTensor` compilation to handle
   output and nociceptor regions in the same tensor block:
   - output neurons: indices `[tissue_end : tissue_end + output_n]`
   - nociceptor neurons: `NeuronaEntrada`, excluded from `procesar()` updates
   - nociceptor→tissue connection: `weight < 0`, `es_input_syn=True`

5. **WebSocket frame** — add `output_grid` (and optionally `label_grid`,
   `nociceptor_grid`) to the frame sent to the client each step.

#### Frontend

6. **Output canvas** — a second `PixelCanvas` alongside the tissue canvas,
   showing output region activation.

7. **Label/nociceptor overlay** — optional: show label signal on output canvas,
   and error pixels on input canvas.

---

### Experiment 1: Supervised Classifier ← **start here**

The simplest ITON case: no `error_diff` computation needed. The supervision
signal is the class label itself, injected directly into the output region as a
`NeuronaEntrada`. The nociceptor is implicit — wrong output neurons have low
tension, so they don't reinforce; only the correct output neuron, if it fires, does.

**Signal flow:**

```
input (pattern A or B)
  ↓  fixed connection
tissue (frozen daemon)
  ↓  learned connection  (tissue→output weights)
output (N neurons, one per class)

label source → output[correct_class] = 1, others = 0  (injected as NeuronaEntrada)
```

The tissue→output weights learn via Hebbian rule:
`dW = lr * tension_output * (tissue_activation - weight)`

- When class A shown → label injects output[A]=1 → output[A] fires → tension > 0
  → weights from active tissue neurons to output[A] strengthen.
- output[B] is clamped to 0 by the label → no tension → no update.
- After training: tissue pattern for A strongly drives output[A] and not output[B].

**Why this before Input Mimic:**
- No `error_diff` computation — no need to compare two grids
- The label source is mechanically identical to the existing `ascii` input source
- It directly validates the tissue→output learning path
- Very similar to Dynamic SOM: same frozen tissue, same Hebbian rule, just one
  extra region downstream

**Config template sketch (`supervised_classifier.json`):**

```json
{
  "description": "Supervised classifier: label signal clamps output neurons to ground truth; tissue→output weights learn Hebbianly.",
  "regions": [
    { "id": "input", "grid": { "width": 20, "height": 20 },
      "source": { "type": "ascii", "text": "HALF_TOP,HALF_BOT",
                  "frames_per_char": 15,
                  "noise": { "background": 0.05, "shift": false } } },
    { "id": "tissue", "grid": { "width": 50, "height": 50 },
      "wiring": { "deamon": { "shape": "square",
                              "excitatory": { "offset": 1, "noise": 0.5, "weights": [1,1,1] },
                              "inhibitory": { "offset": 6, "noise": 0.5, "sectors": 12,
                                              "weights": [1,1,1,1,1,1,1,1,1,1,1,1] } },
                  "dendrite_exc_weight": 0.9, "dendrite_inh_weight": -1,
                  "process_mode": "avg_vs_avg_normalized" } },
    { "id": "output", "grid": { "width": 1, "height": 2 },
      "source": { "type": "label", "follows": "input" } }
  ],
  "connections": [
    { "from": "input",  "to": "tissue", "type": "full", "weight": 0.35,
      "density": 0.4, "learning": { "rate": 0.0 } },
    { "from": "tissue", "to": "output", "type": "full", "weight": 0.5,
      "density": 1.0, "learning": { "rate": 0.02 } }
  ]
}
```

**Implementation order:**
1. Output region in `Experiment` + `BrainTensor`
2. `label` source type — N `NeuronaEntrada` neurons, one per class
3. `output_grid` in WebSocket frame
4. Output canvas in the UI
5. `supervised_classifier.json` config template
6. Validate: after N steps, each output neuron responds selectively to one input class

---

### Experiment 2: Input Mimic (unsupervised)

After the supervised classifier is validated, move to unsupervised reconstruction.
The label source is replaced by an `error_diff` nociceptor that computes
`diff(input_clean, output)` and inhibits the tissue neurons responsible for
reconstruction errors. The output region uses `"activation": "soft"` so it
produces a continuous reconstruction signal (not binary), which the `error_diff`
nociceptor can meaningfully diff against the input.

**Requires:** `error_diff` source, nociceptor→tissue inhibitory connection,
`"activation": "soft"` on the output region.

---

### Experiment 3: Output with lateral connections (Level 2)

After Input Mimic is validated, add a daemon connectome to the output region so it
self-organizes topographically — a second SOM layer that compresses the tissue
representation.

---

Any metrics obtained in Stage 3 feed into Stage 4 (Tuning).

---

## Stage 4: Tuning

**Status:** Planned.

**Objective:** Optimize connectomes and system parameters using
systematic search and selection techniques.

**Tools:**

| Tool | Purpose |
|------|---------|
| **Genetic algorithms** | Vary connectomes (masks) and select by metrics |
| **scikit-learn** | Compare network proposals, select by merit (GridSearchCV, pipelines) |

**Parameters to optimize:**

- Number of neurons, synapses, and dendrites
- Excitatory zone, gap, and inhibitory zone (`E G I`)
- Density of each zone (`DE DI`)
- Synapse location relative to the axon
- Axon position relative to the center of mass of connections

**Metrics to refine (must be mature for this stage):**

- Number of daemons
- Noise level
- Formation speed from noise
- Stability of each candidate

**Resumption:** The Daemons laboratory with these optimization tools.

---

## Stage 5: Motor Agents

**Status:** Planned (long-term horizon).

**Objective:** Take the connectionist model into a simulated world where
an agent moves intelligently without language.

**Prerequisite:** Stage 4 (Tuning) must be complete — finding the right connectome
parameters for a body-coupled system is a search problem that requires genetic
algorithm infrastructure.

### Experiment: Simple Living Creature

**Goal:** A simulated agent that moves using motor outputs, learns to flee pain
via nociceptors, and develops avoidance behavior without explicit reward.

**Architecture:**
- Input: sensory readings from a simulated body (proximity, contact)
- Processing: daemon tissue (connectome tuned in Stage 4)
- Output: motor signals (direction, intensity) interpreted by a physics simulation
- Nociceptor: activated by contact/damage events in the simulation

**Hypothesis:** Consistent nociceptive inhibition during harmful contacts will
progressively suppress the motor patterns that led to them — avoidance behavior
without a reward function.

**Pleasure:** Not explicitly modeled. If it emerges, it will appear as reduced
nociceptive activity in states the agent reaches repeatedly without pain —
consistent with the biological view that pleasure has no dedicated receptor,
only the progressive silencing of pain signals.

**Models to emulate:**
- ***Aplysia californica***: sea slug studied by Eric Kandel;
  nervous system with ~20,000 neurons, which enabled the discovery of
  synaptic mechanisms of learning and memory
- **Zebrafish** (*Danio rerio*): transparent and relatively simple
  nervous system; spinal motor circuits well characterized for locomotion

**Scientific inspiration:**

- Kandel, E. R. (2001). *The Molecular Biology of Memory Storage: A
  Dialogue Between Genes and Synapses*. Nobel Lecture.
- Computational models of spinal locomotor circuits in zebrafish
  (eLife, 2021; Nature Neuroscience, 2023).

---

## Future lines (no stage assigned)

| Line | Description |
|------|-------------|
| Musical generation | Neuronal model that synthesizes music from competitive exclusion dynamics |
| Image generation | Explore the connection between SOMs, Deep Dream, and the connectionist model |
| 3D models | Extend the 2D tissue to 3D neuronal volumes |
| Language | Approach language modeling from the connectionist perspective |

---

← Back to [README](../README.md)

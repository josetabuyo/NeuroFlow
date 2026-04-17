# Roadmap

The stages follow a logical order: each one builds on the previous one.
The ultimate goal is an intelligent motor agent that operates without language,
emulating simple creatures such as *Aplysia* or the zebrafish.

---

## Summary

```
Stage 1   Stage 2       Stage 3              Stage 4      Stage 5
Daemon  → Dynamic SOM → Motor/Nociceptor  → Tuning     → Motor Agents
(✓)       (next)        (theoretical)       (genetic)    (simulation)
```

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

## Stage 2: Dynamic SOM

**Status:** Essentially complete. Pending: metric formalization and tuning.

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

### Pending before closing Stage 2

- Formalize metrics: center-of-mass displacement, daemon bias index per pattern
- Tuning: optimal `lr_input`, `density`, `frames_per_char` for clean separation
- Document the similarity-gradient effect quantitatively

---

## Stage 3: Motor & Nociceptor

**Status:** Planned.

**Objective:** Add output (motor) and error-signal (nociceptor) regions to the
connectionist model. Build progressively from the simplest useful case to a
simulated living creature.

---

### Architecture: three regions

```
Input region  →  Processing region  →  Output (motor) region
                      ↓
              Nociceptor region  (error signal, negative weight)
```

- **Input region:** existing NeuronaEntrada layer
- **Processing region:** existing daemon tissue (lateral exc/inh connectome, frozen)
- **Output region:** new — motor neurons that read from the processing layer
- **Nociceptor region:** new — receives error signal, feeds back with negative weight

---

### Motor region — two levels of complexity

#### Level 1 (implement first): Simple output layer
- No lateral connections between output neurons
- Each output neuron connects to the processing region only (distant connections)
- Fully connected by default, with parametrizable `density`
- Equivalent to the output layer of a simple feedforward net
- Weights learned via the same Hebbian mechanism, same `lr_output` multiplier

#### Level 2 (implement after Level 1 is validated): Output layer with lateral connections
- Output neurons have excitatory-center / inhibitory-surround lateral connections
- This allows the output layer to self-organize topographically as well
- Same mask system as the processing region

---

### Nociceptor region

The nociceptor is a region that carries an **error signal** — the difference between
what was expected and what was produced. It is not a special mechanism; it is another
input region with **negative dendrite weight**, so it inhibits rather than excites.

- Acts exactly like the input region, but with negative weight
- Provides gradient-free error feedback: active nociceptor pixels suppress the
  neurons that fired when they should not have

**No special "pleasure" connections exist in biology** — pleasure is the absence of
nociceptive inhibition, or the reduction of it. This may emerge naturally from the
model without explicit implementation.

---

### Experiment 1: Input Mimic

**Config template name:** `Input Mimic`

**Goal:** Train a system that reproduces its input at the output layer,
filtering noise as a spontaneous effect of its operation.

**Architecture:**
```
Input (noisy pattern)
  ↓ input dendrite (positive weight)
Processing region (daemon tissue, frozen recurrent weights)
  ↓ output dendrite (learned)
Output region (no lateral connections)
  ↓ compare with clean input
Nociceptor region ← diff(input, output): pixels that differ = 1
  ↓ nociceptor dendrite (negative weight, same as input dendrite path)
Processing region (inhibitory feedback)
```

**Why this is interesting:**
- The system learns to reconstruct its input without supervised labels
- Noise filtering emerges spontaneously: noisy input → stable daemon state → clean output
- The nociceptor provides a local error signal without backpropagation
- Has direct utility: denoising, pattern completion, anomaly detection

**Implementation order:**
1. Add output region to `Experiment` class (Level 1: no lateral connections, parametrizable density)
2. Add nociceptor region (error diff projected as negative-weight input)
3. Create `input_mimic.json` config template
4. Add output grid visualization to the UI (second canvas or overlay)
5. Validate: noisy input → reconstructed clean output after training

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

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

**Status:** Next.

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

---

## Stage 3: Motor & Nociceptor

**Status:** Planned.

**Objective:** Explore two fundamental concepts before moving on to
full motor agents:

### Nociceptor

The nociceptor is the pain receptor. In NeuroFlow, modeling
**nociceptive inhibition** makes it possible to understand how a distributed
system can signal "danger" without a central processor — analogous to
gate control theory in the dorsal horn of the spinal cord.

### Motor

How to interpret the **outputs** of a connectionist system as
motor signals. The goal in this stage is not to solve a specific problem,
but rather:

- To develop abstract theoretical models
- To observe what dynamics emerge
- To define what metrics can be obtained

**Experiment:** *Motor & Nociceptor* — theoretical and abstract models.

Any metrics obtained here will feed into Stage 4 (Tuning).

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

**Plan:**

1. Simulate a virtual world with a simple agent (sphere or minimal object)
2. Publish models that emulate living beings without language:
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

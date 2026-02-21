# Experiments — `backend/experiments/`

Experiments orchestrate the neural model: they decide what is input, what
is output, how to feed the network, and how to read the results.

---

## In one line

> Each experiment uses the Constructor to assemble a Network with Regions,
> then processes it frame by frame via WebSocket.

---

## Structure

All experiments inherit from `base.py`:

```
Experiment (base)
├── setup(config)  → uses Constructor to build Network + Regions
├── step()         → network.process() + returns frame
├── click(x, y)    → manipulates neuron in input region
├── reset()        → resets
└── get_frame()    → network.get_grid()
```

---

## Current experiments

### Deamons Lab (`deamons_lab.py`)

**Stage:** 1 — Finding the Daemon (practically covered)

Connectivity lab where different Daemon masks (`E G I DE DI`) are tested
to observe daemon dynamics: formation, stability, movement, competitive
exclusion, and natural balance.

**Configuration:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `width` | Grid width | 30 |
| `height` | Grid height | 30 |
| `mask` | Mask preset | (first preset) |

**What to observe:**

- Formation of daemons (stable activation bubbles)
- Directional movement of daemons
- Competitive exclusion between nearby daemons
- Natural balance (~50% active neurons)
- Convergence upon external manipulation

Available masks are loaded dynamically from `core/masks.py`.
See [Neural Model](../core/README.md) for the nomenclature.

---

## Planned experiments

| # | Name | Stage | Description |
|---|------|-------|-------------|
| 2 | **Dynamic SOM** | 2 | Self-organizing map with the connectionist model |
| 3 | **Motor & Nociceptor** | 3 | Theoretical models of motor output and nociceptive inhibition |
| 4 | **Tuning** | 4 | Optimization with genetic algorithms and scikit-learn |
| 5 | **Motor Agents** | 5 | Agents in simulated world (Aplysia, zebrafish) |

See [Roadmap](../../docs/STAGES.md) for details on each stage.

---

## How to add an experiment

1. Create a new file in this folder (e.g. `dynamic_som.py`)
2. Inherit from `Experiment` (in `base.py`)
3. Implement `setup()`, `step()`, `click()`, `reset()`
4. Register in `__init__.py`
5. The frontend discovers it automatically via the API

---

← Back to [README](../../README.md)

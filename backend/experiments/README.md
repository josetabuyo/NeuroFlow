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

## Config Reference

The canonical config is a JSON object with `regions[]` and `connections[]`.

### Region fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique region identifier |
| `grid.width` / `grid.height` | int | Neuron grid dimensions |
| `wiring` | object | Intra-region connectivity (makes neurons regular `Neurona`) |
| `source` | object | External injection (makes neurons `NeuronaEntrada`) |
| `threshold` / `umbral` | float | Firing threshold, default `0.0` |
| `process_mode` | string | Tension combination: `min_vs_max` (default), `sum`, `avg_vs_avg`, `avg_vs_avg_normalized`, `pos_vs_neg`, `group_avg` |
| `activation` | string | `"soft"` — maps tension directly to `[0,1]` instead of binary threshold (see below) |
| `tension.function` | object | Composable tension transform: `{"x": N, "x_pow_2": N, "x_pow_3": N, "b": N}` |
| `spiking` | object | Spike-frequency adaptation: `{"up_ticks": N, "down_ticks": N}` |

### `activation: "soft"`

When set on any region with `wiring`, neurons output their surface tension clamped
to `[0, 1]` instead of firing binary 0/1.

```json
{
  "id": "output",
  "grid": { "width": 4, "height": 4 },
  "wiring": { "deamon": { ... } },
  "activation": "soft"
}
```

**Applies to:** any region — tissue, output, nociceptor with wiring.
**Not applicable to:** pure source regions (`NeuronaEntrada`), which always
  take their values from external injection.

**Effect on serialization:** `get_region_frames()` emits 3-decimal floats
  instead of rounded 0/1 for soft regions, so the frontend receives the full
  analog signal.

**Typical use cases:**
- Output region as a continuous classifier score instead of winner-take-all
- Nociceptor region that propagates graded error magnitude rather than spikes
- Any intermediate region where preserving analog information matters

### `wiring` fields

| Field | Description |
|-------|-------------|
| `deamon` | Intra-region daemon wiring (shape, excitatory, inhibitory, fixed) |
| `process_mode` | Overrides the region-level `process_mode` |
| `learning_rate` | Learning rate for intra-region synapses (0 = frozen) |

### `source` types

| `type` | Description |
|--------|-------------|
| `"ascii"` | Rendered text / synthetic patterns (`HALF_TOP`, `HALF_BOT`, …) |
| `"label"` | One-hot class signal (supervised target) |
| `"error_diff"` | `abs(target − output)` diff each step (nociceptor) |
| `"label_mismatch"` | Fires when label ≠ predicted class (nociceptor) |
| `"draw"` | User-painted values, with optional background noise |

### Connection fields

| Field | Description |
|-------|-------------|
| `from` / `to` | Source and destination region IDs |
| `full.weight` | Dendrite weight for the connection |
| `full.density` | Fraction of source neurons each dest neuron connects to |
| `full.learning.rate` | Learning rate for this connection (0 = frozen) |

---

## How to add an experiment

1. Create a new file in this folder (e.g. `dynamic_som.py`)
2. Inherit from `Experiment` (in `base.py`)
3. Implement `setup()`, `step()`, `click()`, `reset()`
4. Register in `__init__.py`
5. The frontend discovers it automatically via the API

---

← Back to [README](../../README.md)

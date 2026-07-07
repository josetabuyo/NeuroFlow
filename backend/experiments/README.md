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
| `delta_weight` | object | Per-neuron excitatory/inhibitory weight totals: `{"excitatory": N, "inhibitory": N}` (see below) |

### `delta_weight`

Declares the **total** excitatory and/or inhibitory weight every neuron in the
region should end up with, regardless of how many dendrites contribute to
each polarity or where they come from (daemon mask, nerve connection, or
both). This is defined to match exactly how `process_mode: "group_avg"`
combines dendrites at runtime — **not** a flat sum across every dendrite:

- Dendrites are split into two buckets per polarity: **local** (intra-region,
  i.e. daemon/mask wiring) and **distant** (cross-region, i.e. a nerve).
- Each bucket's dendrites are **averaged** together (this is what `group_avg`
  actually computes per bucket at runtime).
- The two bucket averages are **added**; `delta_weight` is that sum.

After all wiring for the region is complete, every neuron's dendrites of a
given polarity are rescaled by one shared factor so the local-bucket average
plus the distant-bucket average equals the configured total — preserving
each dendrite's relative weight within its own bucket.

```json
{
  "id": "tissue",
  "grid": { "width": 50, "height": 50 },
  "delta_weight": { "excitatory": 0.6, "inhibitory": -0.7 }
}
```

- A neuron with one excitatory and one inhibitory dendrite (both local) gets
  exactly those totals as its dendrite weights.
- A neuron with 12 equal-weight local inhibitory sectors (a `crown`/`circle`
  mask with `sectors: 12`) lands each sector's weight **on** `-0.7`, not
  `-0.7 / 12` — because `group_avg` averages the 12 sectors, it doesn't sum
  them. If the sectors already averaged to `-0.7` (the pre-`delta_weight`
  convention: every sector gets the flat `deamon.inhibitory.weight`), nothing
  changes at all.
- A neuron that also receives a nerve dendrite of the same polarity gets its
  own **distant** bucket, averaged separately and added to the local bucket —
  so a tissue region with a nerve inserted in one corner keeps a continuous
  excitatory/inhibitory balance across the whole region instead of spiking
  wherever the nerve lands.
- Omitting `excitatory` or `inhibitory` leaves that polarity unscaled.
- Without `delta_weight` at all, dendrite weights are exactly whatever the
  wiring config (`deamon.excitatory.weight`, `nerve.from.weight`, ...) says —
  unchanged from before this field existed.
- **Only exact under `process_mode: "group_avg"`.** Under the default
  `min_vs_max` (competitive max/min across dendrites), only the single
  strongest dendrite of each polarity counts toward tension — `delta_weight`
  still redistributes weights proportionally, but the resulting tension
  won't equal the configured total the way it does under `group_avg`.

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
| `"label_mismatch"` | **Suppressive nociceptor** — fires only when a neuron is active but should NOT be: `relu(activation − target)`. Silent for the correct neuron regardless of whether it is active. |
| `"draw"` | User-painted values, with optional background noise |

#### `label_mismatch` — Suppressive nociceptor

Designed for antagonistic multi-class setups (e.g. L vs T regions). Each nociceptor
neuron has a label (via `neuron_labels` on `label_region`) and fires only when its
corresponding output neuron is active but the current input is a different class.

**Signal formula:** `noci[i] = relu(output[i] − target[i])`

where `target[i] = 1.0` if `neuron_labels[i] == current_char`, else `0.0`.

| Case | Signal |
|------|--------|
| Correct neuron active (label matches, output high) | ≈ 0 — no punishment |
| Correct neuron inactive (label matches, output low) | 0 — **never fires** (prevents death spiral) |
| Wrong neuron active (label doesn't match, output high) | = activation — suppresses it |
| Wrong neuron inactive (label doesn't match, output low) | ≈ 0 — already off |

The death-spiral that the previous `abs()` formula caused: when the correct neuron was
inactive, the symmetric signal would fire the nociceptor → inhibitory connection would
suppress the neuron further → it would become even less likely to activate → runaway
negative reinforcement. The `relu` formula eliminates this by making the nociceptor
purely suppressive.

**Typical wiring:**
```json
{
  "from": "noci_T",
  "to": "tissue",
  "full": { "weight": -0.5 }
}
```

The nociceptor for class T connects with negative weight to the tissue. When showing L,
noci_T fires if T is active → suppresses T-related tissue → L can emerge.
When showing T, noci_T stays silent (T is correct) → no interference.

**Required fields:**

| Field | Description |
|-------|-------------|
| `char_region` | Region ID of the ASCII input (provides `current_char`) |
| `label_region` | Region ID whose `neuron_labels` define expected class per neuron |

#### `"draw"` — user-painted input

A region the user paints on directly (mouse/brush). Rebuilt from scratch every
tick — zero → background noise → loop stamp → live cursor stamp — so nothing
accumulates or leaves a trail once painting stops, except the recorded loop
pattern (below), which is config, not runtime state.

| Field | Description |
|-------|-------------|
| `noise` | Background noise on the region — nested `{"background": n}` or a flat scalar `n`, applied fresh every tick |
| `loop` | Optional record-and-replay pattern. A plain number `N` (legacy) is auto-upgraded to the container below; see [loop reference](#draw-loop) |

<a id="draw-loop"></a>
**`loop` container:**

```json
"loop": {
  "frames": 4,
  "brush": { "radius": 2 },
  "points": [
    { "x": 3, "y": 2 },
    null,
    { "x": 1, "y": 1 },
    null
  ]
}
```

| Field | Description |
|-------|-------------|
| `frames` | Loop length in ticks. The replay phase is `generation % frames` |
| `brush.radius` | Shared brush radius (== brush size // 2) used to reconstruct the painted circle at every phase |
| `points` | One entry per phase, `{"x", "y"}` (brush center) or `null` if nothing was painted at that phase |

**How it works:** while the user holds the cursor down, every tick writes
`points[generation % frames] = {x, y}` — overwriting whatever was there
before, empty or a previous point — and updates `brush.radius`. This happens
on *every* tick the cursor is held (not just when a new paint message
arrives), so holding the mouse still across a full cycle fills in every
phase and replay stays solid instead of flickering once every `frames`
ticks. At render time each phase's stored point (if any) is expanded back
into a filled circle using `brush.radius` — the same offset formula as the
frontend's `generateCircleBrush`.

This lives directly in `source_cfg` (i.e. in the experiment's config JSON,
not a side buffer), so it:
- **Persists across reset** — reset just re-derives from config.
- **Survives unrelated soft config edits** (e.g. tweaking `noise`) — a soft
  update carries `points`/`brush` forward when `frames` is unchanged. Only an
  actual `frames` change starts the pattern over.
- **Can drive a headless replay** with no UI/session involved — the config
  alone fully describes the loop.

If `loop` is absent entirely, nothing is recorded or replayed — draw regions
behave exactly as without the feature.

---

### `orchestrator` — eventos de control por tick

Lista de entradas que disparan acciones en ticks específicos. Se evalúa cada step **antes** del procesamiento neural (excepto `inject`, que corre después).

#### Gradiente escalar — `from / to`

Interpola un valor entre dos ticks y lo asigna a una ruta de config:

```json
{
  "from": {"tick": 0,   "set": "connections[0]['full']['weight'] = 0.0"},
  "to":   {"tick": 100, "set": "connections[0]['full']['weight'] = 1.0"}
}
```

#### One-shot — `at`

Asigna un valor exactamente en un tick:

```json
{"at": {"tick": 50, "set": "connections['output']['full']['weight'] = 0.9"}}
```

Las rutas soportadas en `set`:

| Ruta | Efecto |
|------|--------|
| `connections[N]['full']['weight']` | Peso de conexión full por índice |
| `connections['id']['full']['weight']` | Peso de conexión full por nombre |
| `connections['id']['nerve']['from']['weight']` | Peso de un lado nerve |
| `regions[N]['text']` | Texto de región ascii por índice |

#### Inject — `at` + `inject`

Escribe valores de activación directamente en `brain_tensor.valores` de una región, **después** de `procesar()`. Visible en pantalla en ese mismo tick, y sirve como estado inicial para el tick siguiente.

```json
{"at": {"tick": 0}, "inject": {"region": "tissue", "template": "noise"}}
```

Con rango sostenido usando `tick_end` — repite el patrón cada tick del intervalo:

```json
{"at": {"tick": 0, "tick_end": 20}, "inject": {"region": "tissue", "template": "image", "src": "init_dots.png"}}
```

Los injects con `tick: 0` también se aplican al final del setup, antes del primer step, por lo que el estado inicial (step 0) ya refleja el patrón.

**Campos de `inject`:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `region` | string | ID de la región destino |
| `template` | string | `"noise"` — uniforme random `[0,1]`; `"image"` — PNG/JPG mapeado a activaciones |
| `src` | string | Path a la imagen (solo para `"image"`). Relativo a `backend/configs/`. |

**Semántica de píxel para PNG con canal alpha:**

| Píxel | Efecto |
|-------|--------|
| Blanco opaco (`alpha > 0`, valor `1.0`) | Activa la neurona |
| Negro opaco (`alpha > 0`, valor `0.0`) | Silencia la neurona |
| Transparente (`alpha = 0`) | No toca la neurona — deja el valor existente |

Los JPG sin canal alpha siguen comportándose como antes (sobreescriben toda la región). Un PNG con alpha permite máscaras de 3 estados: activar, silenciar, y dejar libre.

---

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

# NeuroFlow

**A connectionist model of the mind — beyond language.**

A network of artificial neurons seeking the *daemon*: the minimal unit of
distributed processing that emerges without a central observer, inspired by
neuroscience, cellular automata, and the philosophy of consciousness.

---

## In one line

> A 2D tissue of neurons connected by synapses and dendrites where daemons
> compete, stabilize, and self-organize — a connectionist path toward
> mind emulation.

---

## What is NeuroFlow?

NeuroFlow is a connectionist neural automata framework. Each pixel on the
screen is a neuron. Neurons connect to each other through dendrites and
synapses. Behavior emerges from simple local rules, without a central
controller.

```
Synapse (weight ≥ 0)  →  Dendrite (weight ∈ [-1,1])  →  Neuron (active/inactive)
   recognizes pattern       fuzzy AND + inhibition          competitive fuzzy OR
```

The project does not aim to replicate what LLMs already solve (language),
but rather to tackle less explored areas: **movement**, **visual perception**,
and the **depth of reasoning** — what we might informally call *intuition*.

---

## Documentation

The documentation is organized by depth level. Start wherever interests you:

| Document | What you'll find |
|----------|-----------------|
| **[Vision and Philosophy](docs/VISION.md)** | What is a daemon, why no central observer, theoretical inspirations |
| **[Roadmap](docs/STAGES.md)** | The 5 stages of the project: Daemons → SOM → Motor/Nociceptor → Tuning → Agents |
| **[Technical Architecture](docs/ARCHITECTURE.md)** | Stack, class design, API, WebSocket protocol, hosting |
| **[References](docs/REFERENCES.md)** | Complete bibliography with citations: Dennett, Hawkins, Kohonen, Kandel (*Aplysia*) and more |
| **[About the Author](docs/AUTHOR.md)** | José Miguel Tabuyo — career, motivation, and dedication |
| **[Decisions (ADR)](docs/decisions/)** | Architecture Decision Records: why WebSocket, why PyTorch, why React, etc. |

### Code-level documentation

| Document | What you'll find |
|----------|-----------------|
| **[Neural Model](backend/core/README.md)** | How Synapse, Dendrite, Neuron, and Network work in code |
| **[Experiments](backend/experiments/README.md)** | What each experiment does, how it's configured, what to observe |

---

## Config Reference

NeuroFlow experiments are defined by a JSON config with two top-level arrays: `regions` and `connections`.

### Regions

Each region entry:

```json
{
  "id": "tissue",
  "grid": { "width": 50, "height": 50 },
  "wiring": { ... },
  "source": { ... },
  "threshold": 0.0,
  "spiking": { ... }
}
```

- **`id`** (string, required): unique identifier
- **`grid`** (object, required): `{ "width": N, "height": N }` — grid dimensions. **Hard param** (requires Refresh).
- **`threshold`** (float, default 0.0): firing threshold applied to all non-input neurons in this region — a neuron fires only when its tension exceeds this value. **Soft param**. Alias: `"umbral"` (legacy).
  - Tissue regions with lateral daemon wiring self-balance around 0, so threshold=0 works fine.
  - Regions without lateral wiring (e.g. output) receive a residual positive signal (~0.35) from silent sources due to the synapse formula `1-|w-input|`. Set `threshold > 0` (e.g. 0.5) to gate this out.
  - Input neurons (`source` regions) are always exempt — their values are set externally.
- **`wiring`** (object): makes this a processing region. **Topology fields are Hard; behavior fields are Soft** (see below).
- **`source`** (object): makes this an input region (NeuronaEntrada). **Hard param** to change type.
- **`spiking`** (object): spike-frequency adaptation. **Soft param**.

A region can have `wiring` only, `source` only, or both.

#### `wiring` object

Topology (Hard — requires Refresh):
- **`mask`** (string): named wiring preset. Mutually exclusive with `deamon`.
- **`deamon`** (object): daemon-style dynamic wiring. Mutually exclusive with `mask`.
  - `shape`: `"square"` | `"ring"`
  - `excitatory`: `{ "offset": N, "noise": 0–1, "weights": [w1, w2, ...] }`
  - `inhibitory`: `{ "offset": N, "noise": 0–1, "sectors": N, "weights": [...] }`
- **`dendrite_exc_weight`** (float): global excitatory dendrite weight. Hard param.
- **`dendrite_inh_weight`** (float, negative): global inhibitory dendrite weight. Hard param.

Behavior (Soft — applies live without Refresh):
- **`process_mode`** (string): how dendrite values combine into tension.
  - `"min_vs_max"` (default): best excitatory - best inhibitory
  - `"avg_vs_avg"`: average excitatory - average inhibitory
  - `"avg_vs_avg_normalized"`: ratio exc/inh — balance matters, not scale
  - `"sum"`: all dendrites summed and clamped
  - `"group_avg"`: separate averages for local-exc, local-inh, cross-exc, cross-inh
- **`tension_function`** (object): polynomial transform applied to tension before activation. Soft param.
  - Keys: `"x"`, `"x_pow_2"`, `"x_pow_3"`, etc.
  - Each key coefficient multiplies that power of tension. `{ "x": 1, "x_pow_3": 20, "x_pow_2": 3 }` = `t + 20t³ + 3t²`
  - All-zero or omitted: tension passes through unchanged.
- **`learning_rate`** (float): intra-region Hebbian learning rate. Soft param.

#### `source` object

```json
{ "type": "ascii", "text": "ABC", "frames_per_char": 15, "font": "press_start_2p", "font_size": 10,
  "noise": { "background": 0.05, "shift": false, "inter_char": false } }
```

- **`type`** (string): `"ascii"` | `"error_diff"` | `"label"`. Hard param to change type.
- For `type: "ascii"` — all fields below are **Soft params**:
  - **`text`** (string): characters to cycle through, or synthetic pattern names comma-separated (`"HALF_TOP,HALF_BOT,BARS_H,BARS_V,DOT_TL,DOT_BR"`)
  - **`frames_per_char`** (int): steps to display each character
  - **`font`** (string): font identifier (e.g. `"press_start_2p"`)
  - **`font_size`** (int): font size in pixels
  - **`noise.background`** (float 0–1): random pixel flip probability
  - **`noise.shift`** (bool): random 1-pixel shift each frame
  - **`noise.inter_char`** (bool): insert a blank/noise frame between characters
- For `type: "error_diff"`:
  - **`target`** (string): region id to compare against (or `"label"`)
  - **`diff_mode`** (string): `"abs"` | `"relu"`

#### `spiking` object (Soft param)

```json
{ "up_ticks": 5, "down_ticks": 5 }
```

- **`up_ticks`**: max consecutive active steps before forced rest
- **`down_ticks`**: refractory period length in steps

### Connections

```json
{
  "from": "tissue",
  "to": "output",
  "type": "full",
  "weight": 0.5,
  "density": 0.4,
  "learning": { "rate": 0.01, "exclude_range": [0.4, 0.6] }
}
```

- **`from`** / **`to`** (string): region ids. **Hard param**.
- **`type`** (string): `"full"` | `"portion"`. **Hard param**.
- **`weight`** (float): initial dendrite weight for this connection. **Hard param**.
- **`density`** (float 0–1): fraction of source neurons each destination samples. **Hard param**.
- **`portion`** (array `[rows, cols]`): spatial subdivision for `type: "portion"`. **Hard param**.
- **`learning.rate`** (float): Hebbian learning rate for this cross-region connection. **Soft param**.
- **`learning.exclude_range`** (array `[lo, hi]`): skip weight updates for weights in this range. **Soft param**.

### Soft vs Hard params summary

**Soft** (apply live, no rebuild):
- `threshold` (alias `umbral`), `process_mode`, `tension_function`, `wiring.learning_rate`
- `source.text`, `source.font`, `source.font_size`, `source.frames_per_char`, `source.noise.*`
- `spiking.up_ticks`, `spiking.down_ticks`
- `connections[].learning.rate`, `connections[].learning.exclude_range`

**Hard** (require Refresh Experiment / full rebuild):
- `grid.width`, `grid.height`
- `wiring.mask`, `wiring.deamon`, `wiring.dendrite_exc_weight`, `wiring.dendrite_inh_weight`
- `source.type`
- `connections[].from`, `connections[].to`, `connections[].type`, `connections[].weight`, `connections[].density`
- Adding or removing regions

---

## Stack

| Layer | Technology | Hosting |
|-------|-----------|---------|
| Backend | Python 3.11+ / FastAPI / WebSocket / PyTorch | Render.com (free) |
| Frontend | Vite / React 19 / TypeScript / HTML5 Canvas | Vercel (free) |

```
Frontend (React + Canvas)  ←WebSocket→  Backend (FastAPI)
         UI                                Network → Neuron → Dendrite → Synapse
```

---

## Fast Start

```bash
./start.sh
```

Starts backend (`:8501`) and frontend (`:5173`) in parallel.
Open **http://localhost:5173** in the browser.

### From scratch

```bash
# Backend (port 8501)
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8501

# Frontend (port 5173) — in another terminal
cd frontend
npm install
npm run dev
```

### Tests

```bash
# Unit tests (backend)
cd backend && pytest -v

# E2E (frontend + backend, Playwright)
cd frontend
npx playwright install        # first time only
npm run test:e2e              # headless
npm run test:e2e:ui           # interactive mode
```

---

## Origin

This project evolved from [RedJavaScript](https://github.com/), a
100% in-browser implementation. NeuroFlow separates frontend (visualization)
from backend (computation), allowing it to scale and deploy as a web service.

---

## License

*To be defined.*

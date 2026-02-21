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
| **[References](docs/REFERENCES.md)** | Complete bibliography with citations: Dennett, Hawkins, Kohonen, Kandel and more |
| **[About the Author](docs/AUTHOR.md)** | José Miguel Tabuyo — career, motivation, and dedication |

### Code-level documentation

| Document | What you'll find |
|----------|-----------------|
| **[Neural Model](backend/core/README.md)** | How Synapse, Dendrite, Neuron, and Network work in code |
| **[Experiments](backend/experiments/README.md)** | What each experiment does, how it's configured, what to observe |

---

## Stack

| Layer | Technology | Hosting |
|-------|-----------|---------|
| Backend | Python 3.11+ / FastAPI / WebSocket / NumPy | Render.com (free) |
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

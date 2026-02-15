# NeuroFlow

Framework de autómatas neuronales conexionistas.

Una grilla de neuronas conectadas por dendritas y sinapsis que reproduce
comportamientos conocidos (autómatas celulares) y luego los trasciende
con aprendizaje emergente.

## Concepto

Cada pixel de la pantalla es una **neurona**. Las neuronas se conectan entre sí
mediante **dendritas** (ramas de entrada) y **sinapsis** (conexiones pesadas).
El comportamiento emerge de reglas locales simples, no de un controlador central.

```
Sinapsis (peso ≥ 0)  →  Dendrita (peso ∈ [-1,1])  →  Neurona (activa/inactiva)
   reconoce patrón         AND fuzzy + inhibición         OR fuzzy competitivo
```

Inspirado en teorías de Kohonen (SOMs), Hawkins (HTM), Dennett (consciencia
distribuida) y neurociencias (células de lugar/grilla).

## Stack

| Capa | Tecnología | Hosting |
|------|-----------|---------|
| Backend | Python 3.11+ / FastAPI / WebSocket / NumPy | Render.com (free) |
| Frontend | Vite / React 19 / TypeScript / HTML5 Canvas | Vercel (free) |

## Experimentos

| # | Nombre | Descripción | Estado |
|---|--------|-------------|--------|
| 0 | **Von Neumann** | Autómata elemental 1D (Wolfram Rules 111, 30, 90, 110) implementado con sinapsis/dendritas/neuronas | Próximo |
| 1 | **Conway** | Game of Life con vecindad Moore (8 vecinos) | Planificado |
| 2 | **Aprendizaje** | Entrenamiento Hebbiano + poda sináptica | Planificado |
| 3 | **Kohonen** | Mapas auto-organizados | Planificado |

## Arquitectura

Ver [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) para diagramas y diseño completo.

```
Frontend (React + Canvas)  ←WebSocket→  Backend (FastAPI)
         UI                                Red → Neurona → Dendrita → Sinapsis
```

## Origen

Este proyecto evoluciona de [RedJavaScript](/Users/josetabuyo/Personal/RedJavaScript),
una implementación 100% en navegador. NeuroFlow separa frontend (visualización)
de backend (cómputo), permitiendo escalar y desplegar como servicio web.

## Quick Start

```bash
# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## Desarrollo

```bash
# Tests
cd backend && pytest -v

# Frontend dev
cd frontend && npm run dev
```

# 0001 — Web Application, Not Jupyter Notebooks

**Status:** Accepted

**Date:** 2025

## Context

NeuroFlow needs real-time visualization of a 2D neuronal tissue running
at 10–30 fps, with user interaction (clicking neurons, changing wiring
on the fly). The natural choice in the Python/ML ecosystem would be
Jupyter Notebooks.

## Decision

Build NeuroFlow as a web application (FastAPI backend + React frontend)
instead of using Jupyter Notebooks.

## Consequences

- **Deployable**: accessible from any browser without local installation.
- **Multi-user**: multiple people can observe the same simulation.
- **Real-time**: WebSocket streaming at full frame rate, not limited by
  notebook cell execution.
- **Interactive**: HTML5 Canvas allows pixel-level rendering and click
  interaction that notebooks cannot match.
- **Trade-off**: more infrastructure to maintain (two codebases, hosting,
  CORS, WebSocket protocol).

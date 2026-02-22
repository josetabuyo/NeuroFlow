# 0002 — Separate Frontend and Backend

**Status:** Accepted

**Date:** 2025

## Context

The neural model processes 2500+ neurons per step with tensor operations.
The UI must remain responsive at all times — rendering the grid, handling
clicks, and updating controls. Coupling computation and rendering in a
single process risks blocking the UI during heavy computation.

## Decision

Separate the system into two independent applications:
- **Backend** (Python/FastAPI): neural computation, experiment logic.
- **Frontend** (React/TypeScript): visualization, user interaction.

Communication via WebSocket for real-time frame streaming.

## Consequences

- Neural computation can use GPU without affecting UI responsiveness.
- Each layer scales independently.
- Frontend can be deployed as static files (Vercel), backend as a
  Python service (Render).
- Trade-off: requires WebSocket protocol design and CORS configuration.

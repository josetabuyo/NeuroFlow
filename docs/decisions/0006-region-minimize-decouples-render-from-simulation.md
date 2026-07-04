# 0006 — Region Minimize Decouples Render from Simulation

**Status:** Accepted

**Date:** 2026

## Context

Regions are slices of a single global tensor (`brain_tensor.valores`);
the backend always computes every region on every tick regardless of
what the frontend does with the data. For large grids, the client-side
cost is not the WebSocket transfer but the per-frame canvas draw in
`PixelCanvas` — one full redraw of every pixel of every visible region,
every tick.

There was no way to reduce that client-side render cost for a region
without also affecting the simulation, and no way to free UI screen
space for a region the user is not currently watching.

## Decision

Add a per-region "minimize" toggle, purely in the frontend
(`LayerBox.tsx` / `Scene.tsx`). Minimizing a region:

- Collapses its box to a small square in place.
- Stops mounting `PixelCanvas` for that region entirely, so its
  draw-to-canvas effect does not run while minimized.
- Does **not** touch the backend or the WebSocket payload — the region's
  data keeps arriving and the simulation keeps ticking exactly as before.

Restoring the region remounts `PixelCanvas`, which redraws from the
latest grid state.

## Consequences

- Render cost for minimized regions drops to ~0 (no canvas draw), while
  simulation correctness is unaffected — regions are just tensor views,
  so per-region compute could not be skipped anyway.
- Bandwidth is unchanged: minimizing a region does not stop the backend
  from serializing/sending its frames. If bandwidth becomes the
  bottleneck, a future `render_enabled` flag on `RegionState`
  (see `backend/experiments/experiment.py`) could suppress serialization
  per region — not implemented here, since the goal was screen-render
  cost, not transfer cost.
- Minimized state is local UI state (`Set<string>` in `Scene.tsx`), not
  persisted across reloads or sessions.

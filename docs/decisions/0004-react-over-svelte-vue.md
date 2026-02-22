# 0004 — React Over Svelte/Vue

**Status:** Accepted

**Date:** 2025

## Context

The frontend needs: an HTML5 Canvas for pixel-grid rendering, a sidebar
with controls and dropdowns, and a WebSocket hook for real-time state.
All three major frameworks (React, Svelte, Vue) can handle this.

## Decision

Use React 19 with TypeScript, bundled by Vite.

## Consequences

- Largest ecosystem and community — more resources for troubleshooting.
- Mature TypeScript support out of the box.
- For a Canvas-based app with a sidebar, React's component model is
  simple enough — no need for Svelte's reactivity or Vue's template
  syntax.
- Vite provides instant HMR regardless of framework choice.
- Trade-off: slightly more boilerplate than Svelte for reactive state,
  but negligible for this project's scope.

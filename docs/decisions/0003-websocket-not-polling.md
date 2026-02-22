# 0003 — WebSocket, Not HTTP Polling

**Status:** Accepted

**Date:** 2025

## Context

The simulation produces ~10–30 frames per second. Each frame is a full
grid of neuron values that must reach the frontend for rendering. The
client also needs to send user actions (click, play, pause, step, reset,
reconnect) with minimal latency.

## Decision

Use WebSocket (`ws://`) for all real-time communication between frontend
and backend.

## Consequences

- Continuous bidirectional streaming without per-frame HTTP overhead.
- Client actions (clicks, wiring changes) arrive with sub-millisecond
  latency — no extra round-trip.
- Single persistent connection per session.
- Trade-off: slightly more complex connection management (reconnection
  logic, error handling) compared to simple REST polling.

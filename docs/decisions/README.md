# Architecture Decision Records (ADR)

This folder documents key technical decisions using the
[ADR pattern](https://adr.github.io/) (Michael Nygard, 2011).

Each decision is a short file with: **Status**, **Context**, **Decision**,
and **Consequences**.

## Decisions

| # | Decision | Status |
|---|----------|--------|
| [0001](0001-web-app-not-notebooks.md) | Web application, not Jupyter Notebooks | Accepted |
| [0002](0002-separate-frontend-backend.md) | Separate frontend and backend | Accepted |
| [0003](0003-websocket-not-polling.md) | WebSocket, not HTTP polling | Accepted |
| [0004](0004-react-over-svelte-vue.md) | React over Svelte/Vue | Accepted |
| [0005](0005-pytorch-for-computation.md) | PyTorch for neural computation | Accepted |

## Adding a new ADR

1. Create `NNNN-short-title.md` (next sequential number).
2. Use the format: Status, Context, Decision, Consequences.
3. Add a row to the table above.

---

← Back to [Architecture](../ARCHITECTURE.md) · [README](../../README.md)

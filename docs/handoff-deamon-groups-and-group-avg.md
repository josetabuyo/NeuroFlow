# Handoff — deamon `groups[]` schema + `group_avg` tension engine

Status as of 2026-07-15. **Nothing in this work has been committed yet** — all
changes are uncommitted working-tree edits (see `git status`/`git diff` for
the exact file list, 15 files touched). Full backend suite: 580 passed, 4
skipped (`cd backend && ./venv/bin/python -m pytest -q`). Frontend typechecks
clean (`cd frontend && npx tsc --noEmit`, one pre-existing unrelated error in
`App.tsx:3` about an unused `useMemo` import — not caused by this work).

## What's done (two refactors, in sequence)

### 1. `deamon` wiring: fixed `{excitatory, inhibitory}` dict → dynamic `groups[]` array

- **New schema**: `deamon.groups: [{id, weight?, offset, weights, density?,
  noise?, sectors?, multiplier?, learning?}, ...]`. No fixed `excitatory`/
  `inhibitory` keys, no separate `sign` field — polarity is the sign of
  `weight`; a group without an explicit `weight` defaults structurally
  (unsectored/unmultiplied → `+1.0`, sectored/multiplied → `-1.0`). `shape`
  default changed from `"square"` to `"circle"`.
- **Files**: `backend/core/masks.py` (`compile_deamon_wiring`,
  `resolve_group_weight`, `default_group_weight`), `backend/experiments/experiment.py`
  (`_compile_mask`, `_apply_delta_weight`/`_rebalance_dendrite_group` at the
  time, `_deamon_learning_rates`, `_migrate_config`), frontend
  `types/index.ts` (`DeamonGroup`, `DeamonWiring`), `Sidebar.tsx` CONFIG_ROWS.
- **DB migration**: `backend/scripts/migrate_deamon_groups.py` — one-time
  script, already run against the real `neuroflow.db` (14 experiments, 579
  runs migrated, backup at `backend/data/neuroflow.bak-<timestamp>.db`,
  every row verified old-vs-new mask equivalence before commit). The script
  can't usefully re-run now that the old-format compiler support is gone
  (see its docstring) — it's kept for history/reference, job already done.
- **Two real bugs found by an Opus adversarial review and fixed**: `_migrate_config`
  (which runs on every config load, for pasted/legacy JSON) wasn't converting
  already-`deamon`-wrapped blocks still using the old shape (→ compiled to an
  empty mask), and wasn't defaulting `sectors`/`multiplier` (12/8) when
  converting un-wrapped legacy connections. Both fixed; a related
  `delta_weight` legacy-dict crash was fixed at the same time. See
  `_convert_deamon_groups`/`_convert_delta_weight_entries` in `experiment.py`
  (shared by `_migrate_config` and the migration script — single source of
  truth now, this was the actual root cause of one of the two bugs, since the
  script used to keep its own drifted copy).

### 2. `process_mode="group_avg"` tension: sum-of-4-locality-buckets → group-then-sign average

- **Old formula** (`brain_tensor.py::_compute_tension`): dendrites bucketed
  by locality (`is_loc`/daemon vs `is_inp`/nerve-cross-region) × sign, each
  bucket flat-averaged, all 4 **summed**.
- **New formula**: every `Dendrita` now carries a `grupo_id` string (new
  field, `core/dendrita.py`) identifying which wiring block produced it — one
  deamon `groups[]` entry, or one `nerve`/`full` connection. Tension =
  average dendrites sharing a `grupo_id` into one per-group value → split
  those group-values by sign → average all present positive groups together,
  average all present negative groups together → **diff** (sum, since
  negative side is already negative-valued). One vote per group, not
  weighted by how many raw dendrites a group has.
- **`grupo_id` is currently 100% internal/auto-generated, never in config**:
  `masks.py::compile_deamon_wiring` tags dendrites `"g{index}"` (positional,
  by index in `groups[]` — **does NOT read the group's own declared `id`
  string today**, this is an open issue, see below). Connections get
  synthesized ids like `f"full:{conn_index}"`, `f"nerve:{conn_index}:from"`,
  `f"nerve:{conn_index}:to"`, `nerve_to_deamon` composes
  `f"{conn_grupo_id}:{mask_entry_grupo_id}"`. Threaded through
  `Constructor.aplicar_mascara_2d` / the various `_wire_*` methods in
  `experiment.py` → `ConstructorTensor.compilar` (builds a new `[N, max_dend]`
  tensor, local per-neuron group index, first-seen order) →
  `BrainTensor._compute_tension`.
- **`_rebalance_dendrite_group`/`_apply_delta_weight`** (still in
  `experiment.py`) mirrors the same group-then-average math for
  `delta_weight` rebalancing (sequential `Brain`/`Dendrita` objects, not
  tensors) — kept in lockstep with the tensor formula.
- **Verified impact**: diffed per-step tension arrays for all 14 experiments,
  8 steps, fixed seed, before/after. 10 of 14 changed (any experiment with
  more than one group of the same sign feeding a neuron — daemon with 2+
  same-sign groups, or nerve/full mixing with daemon); 4 stayed byte-identical
  (pure daemon-only, single group per sign — e.g. "Isolated Daemon", "Sharp
  Pow Daemon", "Spiking Daemon", "Isolated Daemon Min vs Max"). This was
  confirmed as an *intentional, wanted* consequence by the user, not a
  regression to avoid.
- New focused tests: `backend/tests/test_brain_tensor.py::TestGroupAvgTension`
  (hand-built `BrainTensor` via `object.__new__`, bypassing
  Constructor/ConstructorTensor, to test the pure group-then-sign math in
  isolation — includes the "two positive groups vote equally regardless of
  dendrite count" test that specifically proves this differs from flat
  averaging).

## Open design discussion — NOT yet implemented, needs a decision before coding

José wants to make `grupo_id` an **explicit, user-authored config field**
(currently 100% auto-generated/hidden) and separately flagged that
`region.delta_weight`'s current shape is wrong. An Opus agent was consulted
(prompt + full response is in this conversation's history) and gave a
concrete recommendation — **not yet implemented**:

1. **Fix the deamon groups[]-id / grupo_id mismatch first.** Discovery: the
   `id` a user already writes in `deamon.groups[].id` is **currently
   decorative** — `compile_deamon_wiring` never reads it, it assigns
   positional `"g{index}"` instead. So before exposing `id` on connections
   "to match how deamon already works," deamon itself needs fixing:
   `compile_deamon_wiring` should use `group.get("id")` as the real
   `grupo_id` (falling back to `"g{index}"` only if absent), so the id the
   user already authors actually does something.

2. **Add optional `id` to connection *endpoints*, not connections.**
   Recommended location: `full.id`, `nerve.from.id`, `nerve.to.id` (a nerve
   connection has two independently-grouped endpoints — `from` feeds INTO the
   `on` region like a `full` connection would, `to` is the `on` region
   feeding OUT to another region, confirmed correct/not-swapped by reading
   `_wire_nerve_from`/`_wire_nerve_to` in `experiment.py:1107-1158`). Omitted
   → keep today's synthesized string (stable, deterministic, nothing
   breaks). Two endpoints declaring the *same* `id` would merge into one
   group — that's the actual payoff of exposing it.

3. **`delta_weight`: do NOT target `grupo_id` directly.**
   `_rebalance_dendrite_group` applies one scale factor across *all*
   same-sign dendrites to hit a single per-polarity target — there's no (and
   per the group_avg design, there *shouldn't be*) a per-individual-group
   target; that would be a different, larger feature. Instead, make polarity
   **explicit** instead of inferred from a sign that has to informally agree
   with a decorative `"id"` label:
   ```
   delta_weight: [{"polarity": "positive", "weight": 0.6}, {"polarity": "negative", "weight": 0.7}]
   ```
   `weight` becomes a **magnitude (≥0)**; `polarity` is the sole source of
   truth for sign. Drop the `"id"` field from delta_weight entries entirely
   (it never did anything real). This touches: `delta_weight_totals` (its
   contract changes from signed-weight to polarity+magnitude),
   `_apply_delta_weight` (its only real caller),
   `scripts/migrate_deamon_groups.py`'s `verify_equivalent` (its other
   caller), `_migrate_config`'s delta_weight conversion path (add a branch:
   old `[{id, weight-signed}]` → new `[{polarity, weight-magnitude}]`,
   dropping `id` — no DB re-migration needed, `_migrate_config` already runs
   on every load and is idempotent), and the frontend `Region.delta_weight`
   type + anywhere the UI authors a signed weight for it.

**None of items 1-3 above are implemented yet.** José was about to decide
whether to have Claude implement all three, or to hand-add `id`s to his
stored configs first and continue after. **Check the live conversation for
which path he chose before touching this** — don't assume.

## Where to pick up

- If resuming this exact conversation (same session): just continue, full
  context is in scope.
- If starting fresh (new session/agent) with only this file: read this
  doc, then read the actual current state of `backend/core/masks.py`
  (`compile_deamon_wiring`), `backend/core/brain_tensor.py`
  (`_compute_tension`'s `group_avg` branch), and
  `backend/experiments/experiment.py` (`_rebalance_dendrite_group`,
  `_apply_delta_weight`, `delta_weight_totals`, and grep `grupo_id=` for
  every construction site) before making any further changes — the "Open
  design discussion" section above describes a plan, not committed code.
- Re-run `cd backend && ./venv/bin/python -m pytest -q` after any change in
  this area — 580 passed / 4 skipped is the known-good baseline.
- Nothing is committed. Before committing, re-confirm with José what should
  be bundled into one commit vs. split (e.g. the two refactors could be two
  separate commits; the open discussion items would be a third).

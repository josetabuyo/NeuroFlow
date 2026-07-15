"""One-time migration: deamon groups[].offset -> groups[].gap (per-group override).

`groups[].offset` was discontinued: a group's starting ring is now always
computed from `gap` (wiring-level default, optionally overridden per group),
never set directly. Converting an explicit offset to an equivalent gap is
not 1:1 — it depends on where the *previous* group ended, so it must be
computed group by group, in order.

For each stored `experiments`/`experiment_runs` config, this walks every
deamon wiring block that has at least one group with an explicit `offset`
and, in order:
  - first group: an explicit offset must be 1 (the only value the new
    resolver can ever produce for a first group, regardless of gap) — the
    key is simply dropped.
  - later groups: computes `needed_gap = old_offset - prev_last_ring - 1`
    (the gap that reproduces the same resolved offset) and sets it as this
    group's own `gap` override, unless it already equals the wiring-level
    default gap (in which case no override is needed at all).

Before writing each row, it recomputes the ring offsets both ways — via a
local reimplementation of the deprecated offset-aware resolver, and via the
current `core.masks._resolve_offsets` on the converted groups — and asserts
they match exactly. Nothing is committed unless every row in the database
verifies equivalent.

Usage: run once, from backend/: ./venv/bin/python scripts/migrate_deamon_group_offset_to_gap.py
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db as dbmod  # noqa: E402
from core.masks import _resolve_offsets  # noqa: E402


def _old_resolve_offsets(groups: list[dict[str, Any]], gap: int) -> list[int]:
    """Reimplementation of the deprecated explicit-offset-aware resolver —
    kept only so this one-time migration can compute equivalent gaps and
    verify old-vs-new equivalence. Mirrors the pre-migration
    core.masks._resolve_offsets exactly (see git history)."""
    resolved: list[int] = []
    prev_last_ring: int | None = None
    for group in groups:
        if "offset" in group:
            off = int(group["offset"])
        elif prev_last_ring is None:
            off = 1
        else:
            off = max(1, prev_last_ring + gap + 1)
        resolved.append(off)
        prev_last_ring = off + len(group["weights"]) - 1
    return resolved


def convert_deamon(deamon: dict[str, Any]) -> tuple[dict[str, Any], list[int] | None]:
    """Returns the converted deamon dict plus the old-algorithm resolved
    offsets list (for verification against the new resolver), or None if
    this deamon uses the named-preset `mask` path — there, `groups` only
    supplies weight overrides (see experiment.py::_compile_mask) and never
    reaches compile_deamon_wiring/offset/gap resolution at all."""
    deamon = copy.deepcopy(deamon)
    if "mask" in deamon:
        return deamon, None

    groups = deamon.get("groups", [])
    wiring_gap = deamon.get("gap", 0)
    old_resolved = _old_resolve_offsets(groups, wiring_gap)

    if not any("offset" in g for g in groups):
        return deamon, old_resolved

    prev_last_ring: int | None = None
    for i, group in enumerate(groups):
        old_off = old_resolved[i]
        if "offset" in group:
            if prev_last_ring is None:
                if old_off != 1:
                    raise ValueError(
                        f"first group resolved offset={old_off} != 1 — "
                        f"cannot represent under the new gap-only model: {group}"
                    )
                del group["offset"]
            else:
                needed_gap = old_off - prev_last_ring - 1
                del group["offset"]
                if needed_gap != wiring_gap:
                    group["gap"] = needed_gap
        prev_last_ring = old_off + len(group["weights"]) - 1

    return deamon, old_resolved


def _all_deamons(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Every deamon dict (region.wiring.deamon or connection.deamon) that
    has a `groups` list."""
    out = []
    for region in config.get("regions", []):
        wiring = region.get("wiring")
        if isinstance(wiring, dict) and isinstance(wiring.get("deamon"), dict):
            out.append(wiring["deamon"])
    for conn in config.get("connections", []):
        if isinstance(conn.get("deamon"), dict):
            out.append(conn["deamon"])
    return out


def convert_config(config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    for region in config.get("regions", []):
        wiring = region.get("wiring")
        if isinstance(wiring, dict) and isinstance(wiring.get("deamon"), dict):
            wiring["deamon"], _ = convert_deamon(wiring["deamon"])
    for conn in config.get("connections", []):
        if isinstance(conn.get("deamon"), dict):
            conn["deamon"], _ = convert_deamon(conn["deamon"])
    return config


def verify_equivalent(old_config: dict[str, Any], new_config: dict[str, Any], label: str) -> None:
    old_deamons = _all_deamons(old_config)
    new_deamons = _all_deamons(new_config)
    if len(old_deamons) != len(new_deamons):
        raise AssertionError(f"{label}: deamon count mismatch ({len(old_deamons)} vs {len(new_deamons)})")
    for old_d, new_d in zip(old_deamons, new_deamons):
        if "mask" in old_d:
            continue  # named-preset path: groups only feed weight overrides, no ring resolution
        groups = old_d.get("groups", [])
        if not groups:
            continue
        old_resolved = _old_resolve_offsets(groups, old_d.get("gap", 0))
        new_resolved = _resolve_offsets(new_d.get("groups", []), new_d.get("gap", 0))
        if old_resolved != new_resolved:
            raise AssertionError(
                f"{label}: resolved offsets mismatch\n"
                f"old={old_resolved} (from {old_d})\n"
                f"new={new_resolved} (from {new_d})"
            )


def main() -> None:
    db_path = dbmod.DB_PATH
    backup_path = db_path.with_name(f"{db_path.stem}.bak-{int(time.time())}{db_path.suffix}")
    shutil.copy2(db_path, backup_path)
    print(f"Backed up {db_path} -> {backup_path}")

    conn = dbmod._connect()
    migrated_experiments = 0
    migrated_runs = 0
    try:
        for exp_id, config_json in conn.execute("SELECT id, config FROM experiments").fetchall():
            old_config = json.loads(config_json)
            new_config = convert_config(old_config)
            verify_equivalent(old_config, new_config, f"experiment {exp_id}")
            if new_config != old_config:
                conn.execute(
                    "UPDATE experiments SET config = ? WHERE id = ?",
                    (json.dumps(new_config), exp_id),
                )
                migrated_experiments += 1

        for run_id, exp_id, config_json in conn.execute(
            "SELECT id, experiment_id, config FROM experiment_runs"
        ).fetchall():
            old_config = json.loads(config_json)
            new_config = convert_config(old_config)
            verify_equivalent(old_config, new_config, f"run {run_id} (experiment {exp_id})")
            if new_config != old_config:
                conn.execute(
                    "UPDATE experiment_runs SET config = ? WHERE id = ?",
                    (json.dumps(new_config), run_id),
                )
                migrated_runs += 1

        conn.commit()
    except Exception:
        conn.rollback()
        print("Migration aborted, no rows committed. DB backup is intact at", backup_path)
        raise
    finally:
        conn.close()

    print(f"Migrated {migrated_experiments} experiments, {migrated_runs} runs.")


if __name__ == "__main__":
    main()

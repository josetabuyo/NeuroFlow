"""One-time migration:

1. Prune `experiment_runs` down to the 10 most recent rows per experiment
   (older run history dropped — reduces the amount of legacy-shaped data
   sitting around unseen).
2. Migrate every remaining `experiments`/`experiment_runs` config through
   `_migrate_config` (deamon groups[], delta_weight → polarity+magnitude,
   already idempotent/canonical-safe) plus an explicit rename of the
   historical deamon groups[] ids "excitatory"/"inhibitory" to the current
   "first_ring"/"second_ring" naming convention (id is purely a positional
   label — it never determines polarity, that's always the sign of the
   group's own `weight`).

Backs up the DB before touching anything, verifies old-vs-new equivalence
(compiled mask sans grupo_id, delta_weight totals) row by row before
committing any change — nothing is written unless every row verifies.

Usage: run once, from backend/: ./venv/bin/python scripts/migrate_delta_weight_polarity_and_prune_runs.py
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
from experiments.experiment import (  # noqa: E402
    _compile_mask,
    _convert_delta_weight_entries,
    _migrate_config,
    delta_weight_totals,
)

KEEP_RUNS_PER_EXPERIMENT = 10

RING_RENAME = {"excitatory": "first_ring", "inhibitory": "second_ring"}


def _rename_ring_ids(config: dict[str, Any]) -> dict[str, Any]:
    """Rename deamon groups[] ids "excitatory"/"inhibitory" -> "first_ring"/
    "second_ring" wherever already present in groups[]-shaped deamon blocks
    (legacy-dict-shaped blocks are handled by _migrate_config's own
    conversion, which already assigns the new names)."""
    config = copy.deepcopy(config)

    def rename_groups(deamon: Any) -> None:
        if not isinstance(deamon, dict):
            return
        for group in deamon.get("groups", []) or []:
            if isinstance(group, dict) and group.get("id") in RING_RENAME:
                group["id"] = RING_RENAME[group["id"]]

    for region in config.get("regions", []):
        wiring = region.get("wiring")
        if isinstance(wiring, dict):
            rename_groups(wiring.get("deamon"))
    for conn in config.get("connections", []):
        rename_groups(conn.get("deamon"))
    return config


def _all_wiring_cfgs(config: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for region in config.get("regions", []):
        wiring = region.get("wiring")
        if isinstance(wiring, dict) and isinstance(wiring.get("deamon"), dict):
            out.append(wiring)
    for conn in config.get("connections", []):
        if isinstance(conn.get("deamon"), dict):
            out.append({"deamon": conn["deamon"]})
    return out


def _all_delta_weights(config: dict[str, Any]) -> list[Any]:
    return [r["delta_weight"] for r in config.get("regions", []) if "delta_weight" in r]


def _mask_sans_grupo_id(mask: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in d.items() if k != "grupo_id"} for d in mask]


def _grupo_ids_renamed_consistently(old_mask: list[dict[str, Any]], new_mask: list[dict[str, Any]]) -> bool:
    if len(old_mask) != len(new_mask):
        return False
    for old_d, new_d in zip(old_mask, new_mask):
        expected = RING_RENAME.get(old_d.get("grupo_id"), old_d.get("grupo_id"))
        if new_d.get("grupo_id") != expected:
            return False
    return True


def verify_equivalent(old_config: dict[str, Any], new_config: dict[str, Any], label: str) -> None:
    old_wirings = _all_wiring_cfgs(old_config)
    new_wirings = _all_wiring_cfgs(new_config)
    if len(old_wirings) != len(new_wirings):
        raise AssertionError(f"{label}: wiring count mismatch ({len(old_wirings)} vs {len(new_wirings)})")
    for old_w, new_w in zip(old_wirings, new_wirings):
        old_mask, old_rw = _compile_mask(old_w)
        new_mask, new_rw = _compile_mask(new_w)
        if old_rw != new_rw:
            raise AssertionError(f"{label}: random_weights flag mismatch")
        if _mask_sans_grupo_id(old_mask) != _mask_sans_grupo_id(new_mask):
            raise AssertionError(f"{label}: compiled mask mismatch (excluding grupo_id)\nold={old_w}\nnew={new_w}")
        if not _grupo_ids_renamed_consistently(old_mask, new_mask):
            raise AssertionError(f"{label}: grupo_id rename inconsistent\nold={old_mask}\nnew={new_mask}")

    old_dws = _all_delta_weights(old_config)
    new_dws = _all_delta_weights(new_config)
    if len(old_dws) != len(new_dws):
        raise AssertionError(f"{label}: delta_weight count mismatch")
    for old_dw, new_dw in zip(old_dws, new_dws):
        # old_dw may still be a legacy shape (dict, or {id, signed weight}
        # list) — convert it the same way convert_config does before
        # comparing totals with the new polarity-shaped list.
        old_totals = delta_weight_totals(_convert_delta_weight_entries(old_dw))
        if old_totals != delta_weight_totals(new_dw):
            raise AssertionError(f"{label}: delta_weight totals mismatch\nold={old_dw}\nnew={new_dw}")


def convert_config(config: dict[str, Any]) -> dict[str, Any]:
    return _rename_ring_ids(_migrate_config(config))


def main() -> None:
    db_path = dbmod.DB_PATH
    backup_path = db_path.with_name(f"{db_path.stem}.bak-{int(time.time())}{db_path.suffix}")
    shutil.copy2(db_path, backup_path)
    print(f"Backed up {db_path} -> {backup_path}")

    conn = dbmod._connect()
    try:
        # ── 1. Prune experiment_runs to the last N per experiment ──────────
        counts_before = dict(
            conn.execute("SELECT experiment_id, COUNT(*) FROM experiment_runs GROUP BY experiment_id").fetchall()
        )
        exp_ids = [row[0] for row in conn.execute("SELECT id FROM experiments").fetchall()]
        pruned_total = 0
        for exp_id in exp_ids:
            keep_ids = [
                row[0]
                for row in conn.execute(
                    "SELECT id FROM experiment_runs WHERE experiment_id = ? ORDER BY id DESC LIMIT ?",
                    (exp_id, KEEP_RUNS_PER_EXPERIMENT),
                ).fetchall()
            ]
            if not keep_ids:
                continue
            placeholders = ",".join("?" * len(keep_ids))
            cur = conn.execute(
                f"DELETE FROM experiment_runs WHERE experiment_id = ? AND id NOT IN ({placeholders})",
                (exp_id, *keep_ids),
            )
            pruned_total += cur.rowcount

        counts_after = dict(
            conn.execute("SELECT experiment_id, COUNT(*) FROM experiment_runs GROUP BY experiment_id").fetchall()
        )

        # ── 2. Migrate every remaining row ──────────────────────────────────
        migrated_experiments = 0
        migrated_runs = 0

        for exp_id, config_json in conn.execute("SELECT id, config FROM experiments").fetchall():
            old_config = json.loads(config_json)
            new_config = convert_config(old_config)
            verify_equivalent(old_config, new_config, f"experiment {exp_id}")
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

    print(f"Pruned {pruned_total} old runs.")
    for exp_id in exp_ids:
        before = counts_before.get(exp_id, 0)
        after = counts_after.get(exp_id, 0)
        if before != after:
            print(f"  experiment {exp_id}: {before} -> {after} runs")
    print(f"Migrated {migrated_experiments} experiments, {migrated_runs} runs.")
    print(f"Backup: {backup_path}")


if __name__ == "__main__":
    main()

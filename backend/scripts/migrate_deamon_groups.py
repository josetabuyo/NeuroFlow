"""One-time migration: deamon wiring {excitatory, inhibitory} dict -> groups[] array.

Converts every stored `experiments` and `experiment_runs` config in
neuroflow.db to the new groups[]-based deamon wiring format (see
core/masks.py and experiments/experiment.py). Before writing each row, it
recompiles the old and new deamon configs through the same runtime path
(_compile_mask) and asserts the resulting masks are identical — nothing is
committed unless every row in the database verifies equivalent.

Usage: run once, from backend/: ./venv/bin/python scripts/migrate_deamon_groups.py

Already run against the real neuroflow.db (14 experiments, 579 runs) — this
script's job is done. Note it can't be meaningfully re-run against genuinely
unmigrated old-shape data anymore: masks.py's compiler no longer understands
the {excitatory, inhibitory} dict shape at all, so verify_equivalent's old-vs-
new mask comparison would compile the old shape to an empty mask and always
raise (fail-safe: aborts + rolls back, never corrupts). Kept for history and
as the reference conversion this repo already applied.
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
    _convert_deamon_groups,
    _convert_delta_weight_entries,
    delta_weight_totals,
)


def convert_config(config: dict[str, Any]) -> dict[str, Any]:
    """Thin driver over experiments.experiment's own legacy-format converters
    — kept as the single source of truth (shared with _migrate_config, which
    runs on every config load) so this script can't silently drift out of
    sync with what the live code actually does."""
    config = copy.deepcopy(config)
    for region in config.get("regions", []):
        wiring = region.get("wiring")
        if isinstance(wiring, dict) and isinstance(wiring.get("deamon"), dict):
            wiring["deamon"] = _convert_deamon_groups(wiring["deamon"])
        if "delta_weight" in region:
            region["delta_weight"] = _convert_delta_weight_entries(region["delta_weight"])
    for conn in config.get("connections", []):
        if isinstance(conn.get("deamon"), dict):
            conn["deamon"] = _convert_deamon_groups(conn["deamon"])
    return config


def _all_wiring_cfgs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Every wiring_cfg-shaped dict (region.wiring, or a synthetic
    {"deamon": ...} for connections) that _compile_mask can be run against."""
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


def verify_equivalent(old_config: dict[str, Any], new_config: dict[str, Any], label: str) -> None:
    old_wirings = _all_wiring_cfgs(old_config)
    new_wirings = _all_wiring_cfgs(new_config)
    if len(old_wirings) != len(new_wirings):
        raise AssertionError(f"{label}: wiring count mismatch ({len(old_wirings)} vs {len(new_wirings)})")
    for old_w, new_w in zip(old_wirings, new_wirings):
        old_result = _compile_mask(old_w)
        new_result = _compile_mask(new_w)
        if old_result != new_result:
            raise AssertionError(f"{label}: compiled mask mismatch\nold={old_w}\nnew={new_w}")

    old_dws = _all_delta_weights(old_config)
    new_dws = _all_delta_weights(new_config)
    if len(old_dws) != len(new_dws):
        raise AssertionError(f"{label}: delta_weight count mismatch")
    for old_dw, new_dw in zip(old_dws, new_dws):
        # old_dw may still be the deprecated dict shape — convert it the same
        # way convert_config does before comparing totals with the new list.
        old_totals = delta_weight_totals(_convert_delta_weight_entries(old_dw))
        if old_totals != delta_weight_totals(new_dw):
            raise AssertionError(f"{label}: delta_weight totals mismatch\nold={old_dw}\nnew={new_dw}")


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

    print(f"Migrated {migrated_experiments} experiments, {migrated_runs} runs.")


if __name__ == "__main__":
    main()

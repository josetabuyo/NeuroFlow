"""One-time migration: delta_weight [{"polarity":..,"weight":..}, ...] list ->
{"positive"?: magnitude, "negative"?: magnitude} dict.

Follow-up to migrate_delta_weight_polarity_and_prune_runs.py — that script
landed the polarity-list intermediate shape; this one lands the final shape
(the parent field is already named `delta_weight`, so the key itself
["positive"/"negative"] is enough, no need for a nested "weight" field).
`_migrate_config` already understands both old shapes and converts them to
this one on every config load, so this script just runs that conversion once
against the stored DB rows so nothing legacy is left sitting there.

Backs up the DB before touching anything, verifies old-vs-new equivalence
(delta_weight_totals) row by row before committing.

Usage: run once, from backend/: ./venv/bin/python scripts/migrate_delta_weight_to_dict_shape.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import db as dbmod  # noqa: E402
from experiments.experiment import _migrate_config, delta_weight_totals  # noqa: E402


def _all_delta_weights(config: dict[str, Any]) -> list[Any]:
    return [r["delta_weight"] for r in config.get("regions", []) if "delta_weight" in r]


def verify_equivalent(old_config: dict[str, Any], new_config: dict[str, Any], label: str) -> None:
    old_dws = _all_delta_weights(old_config)
    new_dws = _all_delta_weights(new_config)
    if len(old_dws) != len(new_dws):
        raise AssertionError(f"{label}: delta_weight count mismatch")
    for old_dw, new_dw in zip(old_dws, new_dws):
        old_totals = delta_weight_totals(
            old_dw if isinstance(old_dw, dict) else {e["polarity"]: e["weight"] for e in old_dw}
        )
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
            new_config = _migrate_config(old_config)
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
            new_config = _migrate_config(old_config)
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
    print(f"Backup: {backup_path}")


if __name__ == "__main__":
    main()

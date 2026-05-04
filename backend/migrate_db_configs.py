"""Migrate all saved configs in the DB from legacy to canonical format.

Run once after the canonical refactor. Safe to re-run — already-canonical
rows are detected and skipped.

Usage:
    cd backend && python migrate_db_configs.py [--dry-run]
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiments.experiment import _to_canonical
from db import DB_PATH


def migrate(dry_run: bool = False) -> None:
    conn = sqlite3.connect(str(DB_PATH))

    rows = conn.execute(
        "SELECT id, experiment, preset_id, config FROM config_snapshots ORDER BY id ASC"
    ).fetchall()

    print(f"Total rows: {len(rows)}")
    migrated = 0
    already_canonical = 0
    errors = 0

    for row_id, experiment, preset_id, config_raw in rows:
        try:
            config = json.loads(config_raw)
        except json.JSONDecodeError:
            print(f"  [ERROR] id={row_id} {experiment}/{preset_id}: invalid JSON")
            errors += 1
            continue

        if "regions" in config:
            already_canonical += 1
            continue

        canonical = _to_canonical(config)
        canonical_raw = json.dumps(canonical)

        if dry_run:
            print(f"  [DRY] id={row_id} {experiment}/{preset_id}: would migrate")
        else:
            conn.execute(
                "UPDATE config_snapshots SET config = ? WHERE id = ?",
                (canonical_raw, row_id),
            )
        migrated += 1

    if not dry_run:
        conn.commit()

    conn.close()

    label = "DRY RUN — " if dry_run else ""
    print(f"\n{label}Results:")
    print(f"  Migrated:          {migrated}")
    print(f"  Already canonical: {already_canonical}")
    print(f"  Errors:            {errors}")
    if dry_run:
        print("\nRun without --dry-run to apply changes.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    migrate(dry_run=dry_run)

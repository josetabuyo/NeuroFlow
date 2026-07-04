"""Experiment persistence — SQLite storage for experiments and their run history."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "neuroflow.db"
CONFIGS_DIR = Path(__file__).parent / "configs"
DATA_DIR = Path(__file__).parent / "data"

logger = logging.getLogger(__name__)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DB_PATH))


def init_db() -> None:
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            config     TEXT    NOT NULL,
            position   INTEGER NOT NULL,
            created_at TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiment_runs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
            config        TEXT    NOT NULL,
            created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_runs_experiment
        ON experiment_runs(experiment_id, id DESC)
    """)
    conn.commit()

    _migrate_legacy_data(conn)

    conn.close()


def _migrate_legacy_data(conn: sqlite3.Connection) -> None:
    """One-time migration from configs/*.json + config_snapshots + session_*.json.

    Only ever populates data when `experiments` is empty — it never touches a
    table that already has rows, so it can never overwrite or duplicate live
    experiment data. On success it drops the legacy config_snapshots table and
    deletes the legacy files it migrated from, so a later restart (even one
    where the user has since deleted every experiment on purpose) can't
    resurrect already-migrated data — there's nothing left to resurrect it from.

    If code shipped before this fix already migrated once (experiments has
    rows) but left config_snapshots lying around, this drops that stale table
    on its own without touching `experiments`, closing the same hole for
    installs that migrated under the old logic.
    """
    has_legacy_table = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='config_snapshots'"
    ).fetchone()
    config_files = sorted(CONFIGS_DIR.glob("*.json")) if CONFIGS_DIR.is_dir() else []
    experiments_count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]

    if experiments_count > 0:
        # Already migrated (possibly by older code that never dropped config_snapshots).
        # Never touch existing experiments — just clear the dangling legacy table so
        # it can't resurrect anything on a future restart.
        if has_legacy_table:
            conn.execute("DROP TABLE config_snapshots")
            conn.commit()
            logger.info("dropped stale legacy config_snapshots table (already migrated)")
        return

    if not has_legacy_table and not config_files:
        return  # fresh install, nothing to migrate

    stem_to_id: dict[str, int] = {}
    runs_migrated = 0

    try:
        position = 0

        for fp in config_files:
            try:
                data = json.loads(fp.read_text())
            except json.JSONDecodeError:
                logger.warning("skip malformed legacy config %s", fp.name)
                continue
            name = data.get("name") or fp.stem
            config = json.dumps(data.get("config", {}))
            cur = conn.execute(
                "INSERT INTO experiments (name, config, position) VALUES (?, ?, ?)",
                (name, config, position),
            )
            stem_to_id[fp.stem] = cur.lastrowid  # type: ignore[assignment]
            position += 1

        if has_legacy_table:
            legacy_experiments = [
                r[0] for r in conn.execute(
                    "SELECT DISTINCT experiment FROM config_snapshots"
                ).fetchall()
            ]
            for stem in legacy_experiments:
                if stem in stem_to_id:
                    continue
                last = conn.execute(
                    "SELECT config FROM config_snapshots WHERE experiment = ? "
                    "ORDER BY id DESC LIMIT 1",
                    (stem,),
                ).fetchone()
                cur = conn.execute(
                    "INSERT INTO experiments (name, config, position) VALUES (?, ?, ?)",
                    (stem, last[0], position),
                )
                stem_to_id[stem] = cur.lastrowid  # type: ignore[assignment]
                position += 1

            rows = conn.execute(
                "SELECT experiment, config, created_at FROM config_snapshots ORDER BY id ASC"
            ).fetchall()
            for experiment, config, created_at in rows:
                exp_id = stem_to_id.get(experiment)
                if exp_id is None:
                    continue
                conn.execute(
                    "INSERT INTO experiment_runs (experiment_id, config, created_at) "
                    "VALUES (?, ?, ?)",
                    (exp_id, config, created_at),
                )
                runs_migrated += 1

        # Session files hold the most recent draft — they win over the file/run config.
        for stem, exp_id in stem_to_id.items():
            session_fp = DATA_DIR / f"session_{stem}.json"
            if not session_fp.is_file():
                continue
            try:
                session_config = json.loads(session_fp.read_text())
            except json.JSONDecodeError:
                continue
            conn.execute(
                "UPDATE experiments SET config = ? WHERE id = ?",
                (json.dumps(session_config), exp_id),
            )

        if has_legacy_table:
            # Fully absorbed into experiment_runs above — drop it so this migration
            # can never re-run and resurrect it, even if `experiments` is later emptied.
            conn.execute("DROP TABLE config_snapshots")

        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("legacy data migration failed — no files were deleted")
        return

    logger.info(
        "migrated %d experiments, %d runs from legacy storage",
        len(stem_to_id), runs_migrated,
    )

    for fp in config_files:
        fp.unlink(missing_ok=True)
    for stem in stem_to_id:
        (DATA_DIR / f"session_{stem}.json").unlink(missing_ok=True)
    (DATA_DIR / "session_last.json").unlink(missing_ok=True)


# ── Experiments (ABM) ──

def _row_to_experiment(row: tuple, include_config: bool) -> dict:
    if include_config:
        id_, name, config, position, created_at, updated_at = row
        return {
            "id": id_, "name": name, "config": json.loads(config),
            "position": position, "created_at": created_at, "updated_at": updated_at,
        }
    id_, name, position, created_at, updated_at = row
    return {
        "id": id_, "name": name, "position": position,
        "created_at": created_at, "updated_at": updated_at,
    }


def list_experiments() -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, name, position, created_at, updated_at "
        "FROM experiments ORDER BY position ASC, id ASC"
    ).fetchall()
    conn.close()
    return [_row_to_experiment(r, include_config=False) for r in rows]


def create_experiment(name: str, config: dict) -> dict:
    conn = _connect()
    max_position = conn.execute(
        "SELECT COALESCE(MAX(position), -1) FROM experiments"
    ).fetchone()[0]
    cur = conn.execute(
        "INSERT INTO experiments (name, config, position) VALUES (?, ?, ?)",
        (name, json.dumps(config), max_position + 1),
    )
    exp_id = cur.lastrowid
    conn.commit()
    row = conn.execute(
        "SELECT id, name, config, position, created_at, updated_at "
        "FROM experiments WHERE id = ?",
        (exp_id,),
    ).fetchone()
    conn.close()
    return _row_to_experiment(row, include_config=True)


def get_experiment(experiment_id: int) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT id, name, config, position, created_at, updated_at "
        "FROM experiments WHERE id = ?",
        (experiment_id,),
    ).fetchone()
    conn.close()
    return _row_to_experiment(row, include_config=True) if row else None


def update_experiment(
    experiment_id: int,
    name: str | None = None,
    config: dict | None = None,
    position: int | None = None,
) -> dict | None:
    conn = _connect()
    fields, values = [], []
    if name is not None:
        fields.append("name = ?")
        values.append(name)
    if config is not None:
        fields.append("config = ?")
        values.append(json.dumps(config))
    if position is not None:
        fields.append("position = ?")
        values.append(position)
    if fields:
        fields.append("updated_at = datetime('now')")
        values.append(experiment_id)
        conn.execute(
            f"UPDATE experiments SET {', '.join(fields)} WHERE id = ?", values,
        )
        conn.commit()
    row = conn.execute(
        "SELECT id, name, config, position, created_at, updated_at "
        "FROM experiments WHERE id = ?",
        (experiment_id,),
    ).fetchone()
    conn.close()
    return _row_to_experiment(row, include_config=True) if row else None


def delete_experiment(experiment_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
    conn.execute(
        "DELETE FROM experiment_runs WHERE experiment_id = ?", (experiment_id,)
    )
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ── Run history ──

def save_run(experiment_id: int, config: dict) -> int:
    """Save a run snapshot. Returns -1 if identical to the last saved run."""
    conn = _connect()

    last = conn.execute(
        "SELECT config FROM experiment_runs WHERE experiment_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (experiment_id,),
    ).fetchone()
    if last and json.loads(last[0]) == config:
        conn.close()
        return -1

    cur = conn.execute(
        "INSERT INTO experiment_runs (experiment_id, config) VALUES (?, ?)",
        (experiment_id, json.dumps(config)),
    )
    run_id = cur.lastrowid

    conn.commit()
    conn.close()
    return run_id  # type: ignore[return-value]


def get_runs(experiment_id: int) -> list[dict]:
    """All executed configs for an experiment, oldest first."""
    conn = _connect()
    rows = conn.execute(
        "SELECT id, config, created_at FROM experiment_runs "
        "WHERE experiment_id = ? ORDER BY id ASC",
        (experiment_id,),
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "config": json.loads(r[1]), "created_at": r[2]}
        for r in rows
    ]


# ── Config reference (last real-world usage lookup) ──

def _find_key_recursive(obj, key: str):
    """DFS for the first dict that has a key matching `key` (case-insensitive).

    Returns that dict (the smallest enclosing object, siblings included) so the
    caller gets a realistic usage snippet instead of just the bare value.
    """
    if isinstance(obj, dict):
        for k in obj:
            if k.lower() == key:
                return obj
        for v in obj.values():
            found = _find_key_recursive(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key_recursive(item, key)
            if found is not None:
                return found
    return None


def find_last_usage(key: str) -> dict:
    """Most recent real config (experiment or run) that used this literal JSON key."""
    key = key.strip().lower()
    if not key:
        return {"found": False, "key": key}

    conn = _connect()
    candidates = [
        (updated_at, name, config, "experiment")
        for _, name, config, updated_at in conn.execute(
            "SELECT id, name, config, updated_at FROM experiments"
        ).fetchall()
    ]
    for experiment_id, config, created_at in conn.execute(
        "SELECT experiment_id, config, created_at FROM experiment_runs"
    ).fetchall():
        name_row = conn.execute(
            "SELECT name FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        name = name_row[0] if name_row else f"experiment #{experiment_id}"
        candidates.append((created_at, name, config, "run"))
    conn.close()

    candidates.sort(key=lambda c: c[0], reverse=True)

    for timestamp, name, config_json, source in candidates:
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError:
            continue
        snippet = _find_key_recursive(config, key)
        if snippet is not None:
            return {
                "found": True,
                "key": key,
                "snippet": snippet,
                "experiment_name": name,
                "timestamp": timestamp,
                "source": source,
            }
    return {"found": False, "key": key}

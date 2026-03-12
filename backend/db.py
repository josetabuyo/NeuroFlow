"""Config persistence — SQLite storage for experiment configurations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "neuroflow.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DB_PATH))


def init_db() -> None:
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS config_snapshots (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment TEXT    NOT NULL,
            preset_id  TEXT    NOT NULL DEFAULT '_default',
            config     TEXT    NOT NULL,
            created_at TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    # Migration: add preset_id if table already existed without it
    cols = [r[1] for r in conn.execute("PRAGMA table_info(config_snapshots)").fetchall()]
    if "preset_id" not in cols:
        conn.execute(
            "ALTER TABLE config_snapshots "
            "ADD COLUMN preset_id TEXT NOT NULL DEFAULT '_default'"
        )
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_exp_preset
        ON config_snapshots(experiment, preset_id, id DESC)
    """)
    conn.commit()
    conn.close()


def save_config(experiment: str, preset_id: str, config: dict) -> int:
    """Save a config snapshot. Returns -1 if identical to the last saved for this preset."""
    conn = _connect()

    last = conn.execute(
        "SELECT config FROM config_snapshots "
        "WHERE experiment = ? AND preset_id = ? ORDER BY id DESC LIMIT 1",
        (experiment, preset_id),
    ).fetchone()
    if last and json.loads(last[0]) == config:
        conn.close()
        return -1

    cur = conn.execute(
        "INSERT INTO config_snapshots (experiment, preset_id, config) VALUES (?, ?, ?)",
        (experiment, preset_id, json.dumps(config)),
    )
    sid = cur.lastrowid

    conn.commit()
    conn.close()
    return sid  # type: ignore[return-value]


def get_latest(experiment: str, preset_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT config FROM config_snapshots "
        "WHERE experiment = ? AND preset_id = ? ORDER BY id DESC LIMIT 1",
        (experiment, preset_id),
    ).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def get_history(experiment: str, preset_id: str) -> list[dict]:
    """All executed configs for an experiment+preset, oldest first."""
    conn = _connect()
    rows = conn.execute(
        "SELECT id, config, created_at FROM config_snapshots "
        "WHERE experiment = ? AND preset_id = ? ORDER BY id ASC",
        (experiment, preset_id),
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "config": json.loads(r[1]), "created_at": r[2]}
        for r in rows
    ]

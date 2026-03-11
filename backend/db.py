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
            config     TEXT    NOT NULL,
            created_at TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_exp
        ON config_snapshots(experiment, id DESC)
    """)
    conn.commit()
    conn.close()


def save_config(experiment: str, config: dict) -> int:
    """Save a config snapshot. Returns -1 if identical to the last saved."""
    conn = _connect()

    last = conn.execute(
        "SELECT config FROM config_snapshots "
        "WHERE experiment = ? ORDER BY id DESC LIMIT 1",
        (experiment,),
    ).fetchone()
    if last and json.loads(last[0]) == config:
        conn.close()
        return -1

    cur = conn.execute(
        "INSERT INTO config_snapshots (experiment, config) VALUES (?, ?)",
        (experiment, json.dumps(config)),
    )
    sid = cur.lastrowid

    conn.commit()
    conn.close()
    return sid  # type: ignore[return-value]


def get_latest(experiment: str) -> dict | None:
    conn = _connect()
    row = conn.execute(
        "SELECT config FROM config_snapshots "
        "WHERE experiment = ? ORDER BY id DESC LIMIT 1",
        (experiment,),
    ).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def get_history(experiment: str) -> list[dict]:
    """All executed configs for an experiment, oldest first."""
    conn = _connect()
    rows = conn.execute(
        "SELECT id, config, created_at FROM config_snapshots "
        "WHERE experiment = ? ORDER BY id ASC",
        (experiment,),
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "config": json.loads(r[1]), "created_at": r[2]}
        for r in rows
    ]

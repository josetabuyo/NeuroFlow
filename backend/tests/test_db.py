"""Tests for experiment persistence (db.py) — CRUD, run history, and legacy migration."""

import json
import sqlite3

import db


def _use_tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "data" / "neuroflow.db")
    monkeypatch.setattr(db, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(db, "DATA_DIR", tmp_path / "data")


# ── ABM (create / read / update / delete) ──

def test_create_and_get_experiment(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    db.init_db()

    created = db.create_experiment("My Experiment", {"regions": []})
    assert created["name"] == "My Experiment"
    assert created["config"] == {"regions": []}
    assert created["position"] == 0

    fetched = db.get_experiment(created["id"])
    assert fetched == created


def test_get_missing_experiment_returns_none(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    db.init_db()
    assert db.get_experiment(999) is None


def test_list_experiments_omits_config_and_orders_by_position(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    db.init_db()

    a = db.create_experiment("A", {"regions": []})
    b = db.create_experiment("B", {"regions": []})

    listed = db.list_experiments()
    assert [e["id"] for e in listed] == [a["id"], b["id"]]
    assert "config" not in listed[0]


def test_update_experiment_name_config_position(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    db.init_db()

    exp = db.create_experiment("Original", {"regions": []})

    updated = db.update_experiment(exp["id"], name="Renamed")
    assert updated["name"] == "Renamed"
    assert updated["config"] == {"regions": []}

    updated = db.update_experiment(exp["id"], config={"regions": ["x"]})
    assert updated["config"] == {"regions": ["x"]}

    updated = db.update_experiment(exp["id"], position=5)
    assert updated["position"] == 5

    assert updated["updated_at"] >= exp["created_at"]


def test_update_missing_experiment_returns_none(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    db.init_db()
    assert db.update_experiment(999, name="x") is None


def test_delete_experiment_cascades_runs(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    db.init_db()

    exp = db.create_experiment("To delete", {"regions": []})
    db.save_run(exp["id"], {"regions": ["run1"]})

    assert db.delete_experiment(exp["id"]) is True
    assert db.get_experiment(exp["id"]) is None
    assert db.get_runs(exp["id"]) == []


def test_delete_missing_experiment_returns_false(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    db.init_db()
    assert db.delete_experiment(999) is False


# ── Run history ──

def test_save_run_and_get_runs_oldest_first(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    db.init_db()

    exp = db.create_experiment("Exp", {"regions": []})
    db.save_run(exp["id"], {"regions": ["v1"]})
    db.save_run(exp["id"], {"regions": ["v2"]})

    runs = db.get_runs(exp["id"])
    assert [r["config"] for r in runs] == [{"regions": ["v1"]}, {"regions": ["v2"]}]


def test_save_run_returns_minus_one_when_identical_to_last(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    db.init_db()

    exp = db.create_experiment("Exp", {"regions": []})
    first_id = db.save_run(exp["id"], {"regions": ["same"]})
    dup_id = db.save_run(exp["id"], {"regions": ["same"]})

    assert first_id != -1
    assert dup_id == -1
    assert len(db.get_runs(exp["id"])) == 1


# ── Legacy migration ──

def test_migration_seeds_experiments_from_config_files(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True)
    (configs_dir / "my_template.json").write_text(json.dumps({
        "name": "My Template", "config": {"regions": ["seed"]},
    }))
    # Non-experiment asset files must survive untouched
    (configs_dir / "init_dots.jpg").write_bytes(b"not-json")

    db.init_db()

    experiments = db.list_experiments()
    assert len(experiments) == 1
    assert experiments[0]["name"] == "My Template"
    assert db.get_experiment(experiments[0]["id"])["config"] == {"regions": ["seed"]}

    # Legacy JSON definition removed, but the image asset stays
    assert not (configs_dir / "my_template.json").exists()
    assert (configs_dir / "init_dots.jpg").exists()


def test_migration_preserves_legacy_run_history_and_orphan_experiment(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True)
    (configs_dir / "known.json").write_text(json.dumps({
        "name": "Known", "config": {"regions": ["file"]},
    }))

    # Pre-populate the legacy config_snapshots table (as if from a real prior run)
    db.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    legacy_conn = sqlite3.connect(str(db.DB_PATH))
    legacy_conn.execute("""
        CREATE TABLE config_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment TEXT NOT NULL,
            preset_id TEXT NOT NULL DEFAULT '_default',
            config TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    legacy_conn.execute(
        "INSERT INTO config_snapshots (experiment, config) VALUES (?, ?)",
        ("known", json.dumps({"regions": ["run1"]})),
    )
    legacy_conn.execute(
        "INSERT INTO config_snapshots (experiment, config) VALUES (?, ?)",
        ("orphan_no_file", json.dumps({"regions": ["orphan_run"]})),
    )
    legacy_conn.commit()
    legacy_conn.close()

    db.init_db()

    experiments = {e["name"]: e for e in db.list_experiments()}
    assert set(experiments) == {"Known", "orphan_no_file"}

    known_runs = db.get_runs(experiments["Known"]["id"])
    assert [r["config"] for r in known_runs] == [{"regions": ["run1"]}]

    orphan_detail = db.get_experiment(experiments["orphan_no_file"]["id"])
    assert orphan_detail["config"] == {"regions": ["orphan_run"]}


def test_migration_session_file_overrides_config(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    (configs_dir / "draft.json").write_text(json.dumps({
        "name": "Draft", "config": {"regions": ["file_default"]},
    }))
    (data_dir / "session_draft.json").write_text(json.dumps({"regions": ["latest_draft"]}))

    db.init_db()

    experiments = db.list_experiments()
    assert len(experiments) == 1
    assert db.get_experiment(experiments[0]["id"])["config"] == {"regions": ["latest_draft"]}
    assert not (data_dir / "session_draft.json").exists()


def test_migration_is_idempotent_across_restarts(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True)
    (configs_dir / "once.json").write_text(json.dumps({
        "name": "Once", "config": {"regions": []},
    }))

    db.init_db()
    db.init_db()  # second boot should not re-run migration or duplicate rows

    assert len(db.list_experiments()) == 1


def test_migration_noop_on_fresh_install(tmp_path, monkeypatch):
    _use_tmp_db(tmp_path, monkeypatch)
    db.init_db()
    assert db.list_experiments() == []

"""Tests for the /api/experiments REST endpoints."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

import db
from api.routes import router


def _client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "data" / "neuroflow.db")
    monkeypatch.setattr(db, "CONFIGS_DIR", tmp_path / "configs")
    monkeypatch.setattr(db, "DATA_DIR", tmp_path / "data")
    db.init_db()

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_create_list_get_experiment(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    resp = client.post("/api/experiments", json={"name": "Exp 1", "config": {"regions": []}})
    assert resp.status_code == 200
    created = resp.json()
    assert created["name"] == "Exp 1"

    resp = client.get("/api/experiments")
    assert resp.status_code == 200
    listed = resp.json()
    assert len(listed) == 1
    assert "config" not in listed[0]

    resp = client.get(f"/api/experiments/{created['id']}")
    assert resp.status_code == 200
    assert resp.json()["config"] == {"regions": []}


def test_get_experiment_404(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.get("/api/experiments/999")
    assert resp.status_code == 404


def test_patch_experiment_rename_and_autosave_config(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    created = client.post("/api/experiments", json={"name": "Old", "config": {}}).json()

    resp = client.patch(f"/api/experiments/{created['id']}", json={"name": "New"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "New"

    resp = client.patch(f"/api/experiments/{created['id']}", json={"config": {"regions": ["x"]}})
    assert resp.json()["config"] == {"regions": ["x"]}


def test_patch_experiment_404(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.patch("/api/experiments/999", json={"name": "x"})
    assert resp.status_code == 404


def test_delete_experiment(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    created = client.post("/api/experiments", json={"name": "Gone", "config": {}}).json()

    resp = client.delete(f"/api/experiments/{created['id']}")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    assert client.get(f"/api/experiments/{created['id']}").status_code == 404


def test_delete_experiment_404(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.delete("/api/experiments/999")
    assert resp.status_code == 404


def test_run_history_save_and_fetch(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    created = client.post("/api/experiments", json={"name": "Exp", "config": {}}).json()
    exp_id = created["id"]

    resp = client.post(f"/api/experiments/{exp_id}/runs", json={"regions": ["v1"]})
    assert resp.status_code == 200
    assert resp.json()["id"] != -1

    # Identical config back-to-back is a no-op save
    resp = client.post(f"/api/experiments/{exp_id}/runs", json={"regions": ["v1"]})
    assert resp.json()["id"] == -1

    resp = client.get(f"/api/experiments/{exp_id}/runs")
    assert resp.status_code == 200
    history = resp.json()["history"]
    assert len(history) == 1
    assert history[0]["config"] == {"regions": ["v1"]}


def test_removed_template_and_session_endpoints_are_gone(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    assert client.get("/api/templates").status_code == 404
    assert client.get("/api/session/last/anything").status_code == 404

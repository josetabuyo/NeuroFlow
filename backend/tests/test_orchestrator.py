"""Tests for the Experiment orchestrator feature."""

import copy
import random

import torch
from experiments.experiment import Experiment, _parse_orch_expr


# ── Config helpers ────────────────────────────────────────────────────────────

def _two_region_config(conn_weight: float = 0.5) -> dict:
    return {
        "regions": [
            {"id": "tissue", "grid": {"width": 8, "height": 8},
             "wiring": {"deamon": {"mask": "simple"}, "process_mode": "min_vs_max"}},
            {"id": "output", "grid": {"width": 2, "height": 2},
             "wiring": {"deamon": {"mask": "simple"}, "process_mode": "min_vs_max"}},
        ],
        "connections": [
            {"from": "tissue", "to": "output",
             "full": {"weight": conn_weight, "density": 1.0, "learning": {"rate": 0.0}}},
        ],
    }


def _dendrite_weights(exp: Experiment, src_id: str, dst_id: str) -> list[float]:
    """Return pesos_dendrita for the src→dst connection (first 4 values)."""
    bt = exp.brain_tensor
    src = exp._regions_by_id[src_id]
    dst = exp._regions_by_id[dst_id]
    NR = bt.n_real
    ms = bt.pesos_sinapsis.shape[1]
    ss = bt.indices_fuente.clamp(0, bt.mascara_entrada.shape[0] - 1)
    di = torch.arange(NR).unsqueeze(1).expand(NR, ms)
    mask = (
        (di >= dst.start) & (di < dst.end)
        & (ss >= src.start) & (ss < src.end)
        & bt.mascara_valida
    )
    return bt.pesos_dendrita[mask][:4].tolist()


# ── _parse_orch_expr ──────────────────────────────────────────────────────────

class TestParseOrchExpr:

    def test_numeric_value(self) -> None:
        segs, val = _parse_orch_expr("connections[2]['full']['weight'] = 0.9")
        assert segs == ["connections", 2, "full", "weight"]
        assert val == 0.9

    def test_integer_value(self) -> None:
        segs, val = _parse_orch_expr("regions[0]['noise'] = 0")
        assert segs == ["regions", 0, "noise"]
        assert val == 0

    def test_string_value_single_quotes(self) -> None:
        segs, val = _parse_orch_expr("regions[0]['text'] = '7'")
        assert segs == ["regions", 0, "text"]
        assert val == "7"

    def test_nested_path(self) -> None:
        segs, val = _parse_orch_expr("connections[4]['nerve']['to']['weight'] = 0.65")
        assert segs == ["connections", 4, "nerve", "to", "weight"]
        assert val == 0.65

    def test_empty_expr(self) -> None:
        segs, val = _parse_orch_expr("")
        assert segs == []
        assert val is None

    def test_no_equals(self) -> None:
        segs, val = _parse_orch_expr("connections[0]['weight']")
        assert segs == []

    def test_boolean_value(self) -> None:
        segs, val = _parse_orch_expr("regions[0]['enabled'] = true")
        assert val is True


# ── Orchestrator: gradient ────────────────────────────────────────────────────

class TestOrchestratorGradient:

    def test_gradient_applies_at_tick_start(self) -> None:
        random.seed(1)
        exp = Experiment()
        cfg = _two_region_config(conn_weight=0.5)
        cfg["orchestrator"] = [
            {"from": {"tick": 0, "set": "connections[0]['full']['weight'] = 0"},
             "to":   {"tick": 100, "set": "connections[0]['full']['weight'] = 1.0"}},
        ]
        exp.setup(cfg)
        exp.step()  # generation 0 fires: t=0, val=0
        weights = _dendrite_weights(exp, "tissue", "output")
        assert all(abs(w) < 1e-5 for w in weights)

    def test_gradient_midpoint(self) -> None:
        random.seed(1)
        exp = Experiment()
        cfg = _two_region_config(conn_weight=0.5)
        cfg["orchestrator"] = [
            {"from": {"tick": 0,   "set": "connections[0]['full']['weight'] = 0"},
             "to":   {"tick": 100, "set": "connections[0]['full']['weight'] = 1.0"}},
        ]
        exp.setup(cfg)
        for _ in range(51):
            exp.step()  # runs generation 0..50
        weights = _dendrite_weights(exp, "tissue", "output")
        # generation=50 → t=50/100=0.5 → val=0.5
        assert all(abs(w - 0.5) < 1e-4 for w in weights)

    def test_gradient_endpoint(self) -> None:
        random.seed(1)
        exp = Experiment()
        cfg = _two_region_config(conn_weight=0.0)
        cfg["orchestrator"] = [
            {"from": {"tick": 0,   "set": "connections[0]['full']['weight'] = 0"},
             "to":   {"tick": 100, "set": "connections[0]['full']['weight'] = 0.8"}},
        ]
        exp.setup(cfg)
        for _ in range(101):
            exp.step()  # generation 100 fires: t=1.0, val=0.8
        weights = _dendrite_weights(exp, "tissue", "output")
        assert all(abs(w - 0.8) < 1e-4 for w in weights)

    def test_gradient_inactive_before_tick_start(self) -> None:
        random.seed(1)
        exp = Experiment()
        cfg = _two_region_config(conn_weight=0.5)
        cfg["orchestrator"] = [
            {"from": {"tick": 50,  "set": "connections[0]['full']['weight'] = 0"},
             "to":   {"tick": 100, "set": "connections[0]['full']['weight'] = 1.0"}},
        ]
        exp.setup(cfg)
        for _ in range(10):
            exp.step()
        weights = _dendrite_weights(exp, "tissue", "output")
        assert all(abs(w - 0.5) < 1e-4 for w in weights), "should not change before tick_start"


# ── Orchestrator: at (one-shot) ───────────────────────────────────────────────

class TestOrchestratorAt:

    def test_at_fires_exactly_once(self) -> None:
        random.seed(1)
        exp = Experiment()
        cfg = _two_region_config(conn_weight=0.5)
        cfg["orchestrator"] = [
            {"at": {"tick": 10, "set": "connections[0]['full']['weight'] = 0.9"}},
        ]
        exp.setup(cfg)
        for _ in range(10):
            exp.step()
        before = _dendrite_weights(exp, "tissue", "output")
        assert all(abs(w - 0.5) < 1e-4 for w in before), "should not have fired yet"
        exp.step()  # generation=10 → fires
        after = _dendrite_weights(exp, "tissue", "output")
        assert all(abs(w - 0.9) < 1e-4 for w in after)

    def test_at_does_not_fire_before_tick(self) -> None:
        random.seed(1)
        exp = Experiment()
        cfg = _two_region_config(conn_weight=0.3)
        cfg["orchestrator"] = [
            {"at": {"tick": 50, "set": "connections[0]['full']['weight'] = 0.9"}},
        ]
        exp.setup(cfg)
        for _ in range(5):
            exp.step()
        weights = _dendrite_weights(exp, "tissue", "output")
        assert all(abs(w - 0.3) < 1e-4 for w in weights)


# ── Orchestrator: get_orchestrator_state ─────────────────────────────────────

class TestOrchestratorState:

    def test_state_empty_when_no_orchestrator(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_two_region_config())
        assert exp.get_orchestrator_state() == []

    def test_state_shows_active_gradient(self) -> None:
        random.seed(1)
        exp = Experiment()
        cfg = _two_region_config()
        cfg["orchestrator"] = [
            {"from": {"tick": 0,   "set": "connections[0]['full']['weight'] = 0"},
             "to":   {"tick": 100, "set": "connections[0]['full']['weight'] = 1.0"}},
        ]
        exp.setup(cfg)
        # After 50 steps, generation=50 → t=0.50, val=0.50
        for _ in range(50):
            exp.step()
        state = exp.get_orchestrator_state()
        assert len(state) == 1
        ev = state[0]
        assert ev["kind"] == "gradient"
        assert abs(ev["value"] - 0.5) < 0.01
        assert abs(ev["progress"] - 0.5) < 0.01

    def test_state_empty_after_gradient_ends(self) -> None:
        random.seed(1)
        exp = Experiment()
        cfg = _two_region_config()
        cfg["orchestrator"] = [
            {"from": {"tick": 0,   "set": "connections[0]['full']['weight'] = 0"},
             "to":   {"tick": 10,  "set": "connections[0]['full']['weight'] = 1.0"}},
        ]
        exp.setup(cfg)
        for _ in range(20):
            exp.step()
        assert exp.get_orchestrator_state() == []


# ── Orchestrator: update_config soft-updates orchestrator ────────────────────

class TestOrchestratorSoftUpdate:

    def test_update_config_replaces_orchestrator(self) -> None:
        random.seed(1)
        exp = Experiment()
        cfg = _two_region_config()
        cfg["orchestrator"] = [
            {"at": {"tick": 999, "set": "connections[0]['full']['weight'] = 0.9"}},
        ]
        exp.setup(cfg)
        assert len(exp._orchestrator) == 1

        new_cfg = copy.deepcopy(cfg)
        new_cfg["orchestrator"] = []
        exp.update_config(new_cfg)
        assert exp._orchestrator == []

"""Tests for the Experiment orchestrator feature."""

import copy
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from experiments.experiment import Experiment, _parse_orch_expr
from core.ascii_renderer import load_image_grid


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


# ── Orchestrator: string connection lookup ────────────────────────────────────

def _nerve_config() -> dict:
    """Minimal config with a nerve connection for string-lookup tests."""
    return {
        "regions": [
            {"id": "tissue", "grid": {"width": 8, "height": 8},
             "wiring": {"deamon": {"mask": "simple"}, "process_mode": "min_vs_max"}},
            {"id": "src", "grid": {"width": 1, "height": 1},
             "source": {"type": "draw"}},
            {"id": "dst", "grid": {"width": 1, "height": 1},
             "process_mode": "min_vs_max",
             "wiring": {"deamon": {"mask": "simple"}}},
        ],
        "connections": [
            {
                "on": "tissue",
                "nerve": {
                    "insertion": {"x": 4, "y": 4},
                    "radius": 3,
                    "from": {"region": "src", "density": 0.5, "weight": -0.1},
                    "to":   {"region": "dst", "density": 0.5, "weight": 0.8},
                },
            },
        ],
    }


class TestOrchestratorStringLookup:
    """connections['region_id'] resolves to the nerve connection involving that region."""

    def test_lookup_by_to_region(self) -> None:
        """connections['dst']['nerve']['to']['weight'] = X updates tissue→dst weight."""
        random.seed(1)
        exp = Experiment()
        cfg = _nerve_config()
        cfg["orchestrator"] = [
            {"at": {"tick": 0, "set": "connections['dst']['nerve']['to']['weight'] = 0.3"}},
        ]
        exp.setup(cfg)
        exp.step()
        weights = _dendrite_weights(exp, "tissue", "dst")
        assert all(abs(w - 0.3) < 1e-4 for w in weights), f"expected 0.3, got {weights}"

    def test_lookup_by_from_region(self) -> None:
        """connections['src']['nerve']['from']['weight'] = X updates src→tissue weight."""
        random.seed(1)
        exp = Experiment()
        cfg = _nerve_config()
        cfg["orchestrator"] = [
            {"at": {"tick": 0, "set": "connections['src']['nerve']['from']['weight'] = -0.5"}},
        ]
        exp.setup(cfg)
        exp.step()
        weights = _dendrite_weights(exp, "src", "tissue")
        assert all(abs(w - (-0.5)) < 1e-4 for w in weights), f"expected -0.5, got {weights}"

    def test_lookup_by_explicit_id(self) -> None:
        """connections['my_nerve']['nerve']['to']['weight'] = X uses explicit id field."""
        random.seed(1)
        exp = Experiment()
        cfg = _nerve_config()
        cfg["connections"][0]["id"] = "my_nerve"
        cfg["orchestrator"] = [
            {"at": {"tick": 0, "set": "connections['my_nerve']['nerve']['to']['weight'] = 0.6"}},
        ]
        exp.setup(cfg)
        exp.step()
        weights = _dendrite_weights(exp, "tissue", "dst")
        assert all(abs(w - 0.6) < 1e-4 for w in weights), f"expected 0.6, got {weights}"

    def test_lookup_gradient_by_region_name(self) -> None:
        """String lookup works in gradient (from/to) entries too."""
        random.seed(1)
        exp = Experiment()
        cfg = _nerve_config()
        cfg["orchestrator"] = [
            {"from": {"tick": 0,   "set": "connections['dst']['nerve']['to']['weight'] = 0.0"},
             "to":   {"tick": 100, "set": "connections['dst']['nerve']['to']['weight'] = 1.0"}},
        ]
        exp.setup(cfg)
        for _ in range(51):
            exp.step()
        weights = _dendrite_weights(exp, "tissue", "dst")
        assert all(abs(w - 0.5) < 1e-4 for w in weights), f"expected 0.5 at midpoint, got {weights}"

    def test_unknown_key_logs_warning(self, caplog) -> None:
        """Unknown string key logs a warning and doesn't crash."""
        import logging
        random.seed(1)
        exp = Experiment()
        cfg = _nerve_config()
        cfg["orchestrator"] = [
            {"at": {"tick": 0, "set": "connections['nonexistent']['nerve']['to']['weight'] = 1"}},
        ]
        exp.setup(cfg)
        with caplog.at_level(logging.WARNING):
            exp.step()
        assert any("nonexistent" in r.message for r in caplog.records)


# ── load_image_grid — alpha channel ──────────────────────────────────────────

class TestLoadImageGridAlpha:

    def _make_png(self, pixels: list[tuple[int, int, int, int]], size=(4, 4)) -> str:
        """Save an RGBA PNG with given flat pixel list and return the path."""
        img = Image.new("RGBA", size)
        img.putdata(pixels * (size[0] * size[1] // len(pixels) + 1))
        path = tempfile.mktemp(suffix=".png")
        img.save(path)
        return path

    def test_jpeg_returns_none_mask(self) -> None:
        img = Image.new("L", (8, 8), 128)
        path = tempfile.mktemp(suffix=".jpg")
        img.save(path)
        grid, mask = load_image_grid(path, 4, 4)
        assert mask is None
        assert grid.shape == (4, 4)
        assert grid.dtype == np.float32

    def test_rgba_png_mask_matches_alpha(self) -> None:
        # Left 2 columns white opaque, right 2 columns transparent
        img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        for y in range(4):
            for x in range(2):
                img.putpixel((x, y), (255, 255, 255, 255))
        p = tempfile.mktemp(suffix=".png")
        img.save(p)
        grid, mask = load_image_grid(p, 4, 4)
        assert mask is not None
        assert mask.shape == (4, 4)
        assert mask[:, 0].all() and mask[:, 1].all()    # left cols → write
        assert not mask[:, 2].any() and not mask[:, 3].any()  # right cols → skip

    def test_white_pixel_value_is_one(self) -> None:
        img = Image.new("RGBA", (4, 4), (255, 255, 255, 255))
        p = tempfile.mktemp(suffix=".png")
        img.save(p)
        grid, mask = load_image_grid(p, 4, 4)
        assert mask is not None and mask.all()
        assert np.allclose(grid, 1.0)

    def test_black_opaque_pixel_value_is_zero(self) -> None:
        img = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
        p = tempfile.mktemp(suffix=".png")
        img.save(p)
        grid, mask = load_image_grid(p, 4, 4)
        assert mask is not None and mask.all()
        assert np.allclose(grid, 0.0)

    def test_fully_transparent_png_mask_all_false(self) -> None:
        img = Image.new("RGBA", (4, 4), (255, 255, 255, 0))
        p = tempfile.mktemp(suffix=".png")
        img.save(p)
        grid, mask = load_image_grid(p, 4, 4)
        assert mask is not None and not mask.any()


# ── Inject image with alpha — selective write ─────────────────────────────────

class TestInjectImageAlpha:

    def _simple_config(self) -> dict:
        return {
            "regions": [
                {"id": "tissue", "grid": {"width": 4, "height": 4},
                 "wiring": {"deamon": {"mask": "simple"}, "process_mode": "min_vs_max"}},
            ],
            "connections": [],
        }

    def test_transparent_pixels_leave_valores_unchanged(self) -> None:
        """Neurons under transparent pixels must not be overwritten."""
        random.seed(1)
        img = Image.new("RGBA", (4, 4), (255, 255, 255, 0))  # fully transparent
        p = tempfile.mktemp(suffix=".png")
        img.save(p)

        exp = Experiment()
        exp.setup(self._simple_config())
        bt = exp.brain_tensor
        rs = exp._regions_by_id["tissue"]
        bt.valores[rs.start:rs.end] = 0.5
        before = bt.valores[rs.start:rs.end].clone()
        exp._apply_orch_inject({"region": "tissue", "template": "image", "src": p})
        after = bt.valores[rs.start:rs.end]
        assert torch.allclose(before, after), "transparent PNG must not change any valor"

    def test_white_opaque_pixels_activate_neurons(self) -> None:
        """Neurons under white opaque pixels should be set to 1.0."""
        random.seed(1)
        img = Image.new("RGBA", (4, 4), (255, 255, 255, 255))
        p = tempfile.mktemp(suffix=".png")
        img.save(p)

        exp = Experiment()
        exp.setup(self._simple_config())
        bt = exp.brain_tensor
        rs = exp._regions_by_id["tissue"]
        exp._apply_orch_inject({"region": "tissue", "template": "image", "src": p})
        after = bt.valores[rs.start:rs.end]
        assert torch.allclose(after, torch.ones_like(after)), "white opaque PNG must set all to 1.0"

    def test_black_opaque_pixels_silence_neurons(self) -> None:
        """Neurons under black opaque pixels should be set to 0.0."""
        random.seed(1)
        img = Image.new("RGBA", (4, 4), (0, 0, 0, 255))
        p = tempfile.mktemp(suffix=".png")
        img.save(p)

        exp = Experiment()
        exp.setup(self._simple_config())
        bt = exp.brain_tensor
        rs = exp._regions_by_id["tissue"]
        exp._apply_orch_inject({"region": "tissue", "template": "image", "src": p})
        after = bt.valores[rs.start:rs.end]
        assert torch.allclose(after, torch.zeros_like(after)), "black opaque PNG must set all to 0.0"

    def test_mixed_alpha_partial_write(self) -> None:
        """Only opaque neurons change; transparent ones keep prior value."""
        random.seed(1)
        # Left half white opaque, right half transparent
        img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        for y in range(4):
            for x in range(2):
                img.putpixel((x, y), (255, 255, 255, 255))
        p = tempfile.mktemp(suffix=".png")
        img.save(p)

        exp = Experiment()
        cfg = self._simple_config()
        exp.setup(cfg)
        bt = exp.brain_tensor
        rs = exp._regions_by_id["tissue"]
        # Set all neurons to 0.5 before inject
        bt.valores[rs.start:rs.end] = 0.5
        # Manually call inject
        exp._apply_orch_inject({"region": "tissue", "template": "image", "src": p})
        after = bt.valores[rs.start:rs.end].view(4, 4)
        assert torch.allclose(after[:, :2], torch.ones(4, 2)), "left half must be 1.0"
        assert torch.allclose(after[:, 2:], torch.full((4, 2), 0.5)), "right half must stay 0.5"

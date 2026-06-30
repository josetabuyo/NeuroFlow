"""Tests for Experiment.update_config — soft vs hard update classification.

Soft updates return True and apply in-place without rebuilding the network.
Hard updates return False and call setup() from scratch.
"""

import copy
import random

import pytest
import torch
from experiments.experiment import Experiment


def _wiring_region(rid: str, w: int, h: int, **wiring: object) -> dict:
    process_mode = wiring.pop("process_mode", "min_vs_max")
    lr = wiring.pop("learning_rate", None)
    exc_w = wiring.pop("dendrite_exc_weight", None)
    inh_w = wiring.pop("dendrite_inh_weight", None)
    deamon: dict = {"mask": "simple"}
    if exc_w is not None:
        deamon["excitatory"] = {"weight": exc_w}
    if inh_w is not None:
        deamon["inhibitory"] = {"weight": inh_w}
    deamon.update(wiring)
    cfg: dict = {"deamon": deamon, "process_mode": process_mode}
    if lr is not None:
        cfg["learning_rate"] = lr
    return {"id": rid, "grid": {"width": w, "height": h}, "wiring": cfg}


def _base_config() -> dict:
    """Minimal canonical config with one wiring region."""
    return {
        "regions": [_wiring_region("tissue", 8, 8)],
        "connections": [],
    }


def _two_region_config() -> dict:
    """Tissue + output with a connection."""
    return {
        "regions": [
            _wiring_region("tissue", 8, 8),
            _wiring_region("output", 4, 4),
        ],
        "connections": [
            {"from": "tissue", "to": "output", "full": {"weight": 0.5, "density": 1.0, "learning": {"rate": 0.0}}},
        ],
    }


def _ascii_config() -> dict:
    """Config with an ASCII source and tissue wiring."""
    return {
        "regions": [
            {"id": "input", "grid": {"width": 6, "height": 6},
             "source": {"type": "ascii", "text": "HALF_TOP,HALF_BOT", "frames_per_char": 5}},
            _wiring_region("tissue", 8, 8),
        ],
        "connections": [
            {"from": "input", "to": "tissue", "full": {"weight": 0.4, "density": 1.0}},
        ],
    }


# ── Soft update: returns True ─────────────────────────────────────────────────

class TestSoftUpdates:

    def test_process_mode_change_is_soft(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["wiring"]["process_mode"] = "sum"
        assert exp.update_config(cfg) is True

    def test_tension_function_change_is_soft(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["tension"] = {"function": {"x": 1, "x_pow_3": 5}}
        assert exp.update_config(cfg) is True

    def test_umbral_change_is_soft(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["umbral"] = 0.3
        assert exp.update_config(cfg) is True

    def test_umbral_propagates_to_brain_tensor(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["umbral"] = 0.42
        exp.update_config(cfg)
        tissue = exp._regions_by_id["tissue"]
        thresholds = exp.brain_tensor.umbrales[tissue.start:tissue.end]
        assert all(abs(v.item() - 0.42) < 1e-6 for v in thresholds)

    def test_connection_learning_rate_change_is_soft(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_two_region_config())
        cfg = copy.deepcopy(exp._config)
        cfg["connections"][0]["full"]["learning"]["rate"] = 0.05
        assert exp.update_config(cfg) is True

    def test_connection_learning_rate_takes_effect(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_two_region_config())
        assert exp.learning_enabled is False

        cfg = copy.deepcopy(exp._config)
        cfg["connections"][0]["full"]["learning"]["rate"] = 0.05
        exp.update_config(cfg)
        assert exp.learning_enabled is True

    def test_intra_region_learning_rate_change_is_soft(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["wiring"]["learning_rate"] = 0.02
        assert exp.update_config(cfg) is True

    def test_intra_region_learning_rate_takes_effect(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["wiring"]["learning_rate"] = 0.02
        exp.update_config(cfg)
        assert exp.learning_enabled is True

    def test_ascii_text_change_is_soft(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_ascii_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["source"]["text"] = "BARS_H,BARS_V"
        assert exp.update_config(cfg) is True

    def test_ascii_text_change_resets_char_index(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_ascii_config())
        # 7 steps: frames_per_char=5 → char switches at step 5, so after 7 steps
        # char_index=1 and frame_in_char=2 (not back to zero yet)
        for _ in range(7):
            exp.step()
        src = exp._ascii_region()
        assert src is not None
        assert src.char_index > 0 or src.frame_in_char > 0

        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["source"]["text"] = "BARS_H,BARS_V"
        exp.update_config(cfg)
        assert src.char_index == 0
        assert src.frame_in_char == 0

    def test_frames_per_char_change_is_soft(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_ascii_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["source"]["frames_per_char"] = 20
        assert exp.update_config(cfg) is True

    def test_noise_background_change_is_soft(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_ascii_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["source"]["noise"] = {"background": 0.1, "shift": False}
        assert exp.update_config(cfg) is True

    def test_process_mode_takes_effect_in_region_specs(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["wiring"]["process_mode"] = "sum"
        exp.update_config(cfg)
        tissue = exp._regions_by_id["tissue"]
        specs = {(s[0], s[1]): s[2] for s in exp.brain_tensor.region_specs}
        assert specs[(tissue.start, tissue.end)] == "sum"

    def test_tension_function_takes_effect_in_region_specs(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["tension"] = {"function": {"x": 2, "x_pow_3": 10}}
        exp.update_config(cfg)
        tissue = exp._regions_by_id["tissue"]
        specs = {(s[0], s[1]): (s[2], s[3]) for s in exp.brain_tensor.region_specs}
        _, fns = specs[(tissue.start, tissue.end)]
        fns_dict = dict(fns)
        assert abs(fns_dict.get("x", 0) - 2.0) < 1e-6
        assert abs(fns_dict.get("x_pow_3", 0) - 10.0) < 1e-6

    def test_multiregion_both_process_modes_update_soft(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_two_region_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["wiring"]["process_mode"] = "avg_vs_avg"
        cfg["regions"][1]["wiring"]["process_mode"] = "sum"
        result = exp.update_config(cfg)
        assert result is True
        tissue = exp._regions_by_id["tissue"]
        output = exp._regions_by_id["output"]
        specs = {(s[0], s[1]): s[2] for s in exp.brain_tensor.region_specs}
        assert specs[(tissue.start, tissue.end)] == "avg_vs_avg"
        assert specs[(output.start, output.end)] == "sum"

    def test_spiking_enable_is_soft(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        assert exp.adaptation_enabled is False
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["spiking"] = {"up_ticks": 3, "down_ticks": 4}
        result = exp.update_config(cfg)
        assert result is True
        assert exp.adaptation_enabled is True
        assert exp.brain_tensor.max_active_steps == 3
        assert exp.brain_tensor.refractory_steps == 4

    def test_network_still_runs_after_soft_update(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        for _ in range(5):
            exp.step()
        gen_before = exp.generation

        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["wiring"]["process_mode"] = "avg_vs_avg"
        exp.update_config(cfg)

        for _ in range(5):
            exp.step()
        assert exp.generation == gen_before + 5


# ── Hard update: returns False ────────────────────────────────────────────────

class TestHardUpdates:

    def test_grid_width_change_is_hard(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["grid"]["width"] = 12
        assert exp.update_config(cfg) is False

    def test_grid_height_change_is_hard(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["grid"]["height"] = 12
        assert exp.update_config(cfg) is False

    def test_add_region_is_hard(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"].append(_wiring_region("extra", 4, 4))
        assert exp.update_config(cfg) is False

    def test_remove_region_is_hard(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_two_region_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"] = [cfg["regions"][0]]
        cfg["connections"] = []
        assert exp.update_config(cfg) is False

    def test_connection_weight_change_is_soft(self) -> None:
        # Weight is not topology — changing it applies to pesos_dendrita without rebuild.
        random.seed(1)
        exp = Experiment()
        exp.setup(_two_region_config())
        gen_before = exp.generation
        cfg = copy.deepcopy(exp._config)
        cfg["connections"][0]["full"]["weight"] = 0.9
        assert exp.update_config(cfg) is True
        assert exp.generation == gen_before  # no reset

    def test_connection_density_change_is_hard(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_two_region_config())
        cfg = copy.deepcopy(exp._config)
        cfg["connections"][0]["full"]["density"] = 0.5
        assert exp.update_config(cfg) is False

    def test_source_type_change_is_hard(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_ascii_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["source"]["type"] = "label"
        assert exp.update_config(cfg) is False

    def test_dendrite_exc_weight_change_is_hard(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["wiring"]["deamon"]["excitatory"] = {"weight": 0.7}
        assert exp.update_config(cfg) is False

    def test_hard_update_resets_generation(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        for _ in range(10):
            exp.step()
        assert exp.generation == 10

        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["grid"]["width"] = 10
        exp.update_config(cfg)
        assert exp.generation == 0

    def test_no_change_is_soft(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        cfg = copy.deepcopy(exp._config)
        assert exp.update_config(cfg) is True

    def test_spiking_disable_is_soft(self) -> None:
        random.seed(1)
        exp = Experiment()
        cfg_with_spiking = _base_config()
        cfg_with_spiking["regions"][0]["spiking"] = {"up_ticks": 3, "down_ticks": 4}
        exp.setup(cfg_with_spiking)
        assert exp.adaptation_enabled is True

        cfg = copy.deepcopy(exp._config)
        del cfg["regions"][0]["spiking"]
        result = exp.update_config(cfg)
        assert result is True
        assert exp.adaptation_enabled is False
        assert exp.brain_tensor.adaptation_enabled is False

    def test_connection_exclude_range_change_is_soft(self) -> None:
        random.seed(1)
        exp = Experiment()
        cfg = _two_region_config()
        cfg["connections"][0]["full"]["learning"] = {"rate": 0.05, "exclude_range": [0.4, 0.6]}
        exp.setup(cfg)

        new_cfg = copy.deepcopy(exp._config)
        new_cfg["connections"][0]["full"]["learning"]["exclude_range"] = [0.3, 0.7]
        assert exp.update_config(new_cfg) is True

    def test_connection_from_change_is_hard(self) -> None:
        random.seed(1)
        exp = Experiment()
        cfg = {
            "regions": [
                _wiring_region("a", 6, 6),
                _wiring_region("b", 4, 4),
                _wiring_region("c", 4, 4),
            ],
            "connections": [
                {"from": "a", "to": "b", "full": {"weight": 0.5, "density": 1.0}},
            ],
        }
        exp.setup(cfg)
        new_cfg = copy.deepcopy(exp._config)
        new_cfg["connections"][0]["from"] = "c"
        assert exp.update_config(new_cfg) is False

    def test_dendrite_inh_weight_change_is_hard(self) -> None:
        random.seed(1)
        exp = Experiment()
        exp.setup(_base_config())
        cfg = copy.deepcopy(exp._config)
        cfg["regions"][0]["wiring"]["deamon"]["inhibitory"] = {"weight": -0.5}
        assert exp.update_config(cfg) is False

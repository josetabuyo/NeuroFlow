"""Tests for deamon wiring compilation and noise semantics.

Validates:
- compile_deamon_wiring emits random_noise on every dendrite
- noise controls per-neuron scaling only (template weights unchanged)
- aplicar_mascara_2d uses random_noise to determine scaling range
- _compute_preview_grid uses the same noise formula as the live network
"""

import random

import pytest

from core.constructor import Constructor
from core.masks import _compute_preview_grid, compile_deamon_wiring


# ── helpers ──────────────────────────────────────────────────────────────────

def _square_wiring(exc_noise=None, inh_noise=None, sectors=4):
    w = {
        "shape": "square",
        "excitatory": {"offset": 1, "weights": [1, 1, 1]},
        "inhibitory": {"offset": 4, "weights": [1, 1, 1], "sectors": sectors},
    }
    if exc_noise is not None:
        w["excitatory"]["noise"] = exc_noise
    if inh_noise is not None:
        w["inhibitory"]["noise"] = inh_noise
    return w


def _build_brain_with_mask(mask, random_weights=True, width=20, height=20):
    c = Constructor()
    brain, _ = c.crear_grilla(
        width=width, height=height, filas_entrada=[], filas_salida=[], umbral=0.0
    )
    c.aplicar_mascara_2d(brain, width, height, mask, random_weights=random_weights)
    return brain


# ── compile_deamon_wiring ─────────────────────────────────────────────────────

class TestCompileDeamonWiringRandomNoise:
    """compile_deamon_wiring must set random_noise on every dendrite."""

    def test_no_noise_defaults_exc_to_0_5(self):
        mask = compile_deamon_wiring(_square_wiring())
        exc = [d for d in mask if d["peso_dendrita"] > 0]
        assert len(exc) == 1
        assert exc[0]["random_noise"] == 0.5

    def test_no_noise_defaults_inh_to_0_5(self):
        mask = compile_deamon_wiring(_square_wiring())
        inh = [d for d in mask if d["peso_dendrita"] < 0]
        assert len(inh) > 0
        assert all(d["random_noise"] == 0.5 for d in inh)

    def test_explicit_exc_noise_propagates(self):
        mask = compile_deamon_wiring(_square_wiring(exc_noise=0.67))
        exc = next(d for d in mask if d["peso_dendrita"] > 0)
        assert exc["random_noise"] == pytest.approx(0.67)

    def test_explicit_inh_noise_propagates(self):
        mask = compile_deamon_wiring(_square_wiring(inh_noise=0.67))
        inh = [d for d in mask if d["peso_dendrita"] < 0]
        assert all(d["random_noise"] == pytest.approx(0.67) for d in inh)

    def test_noise_zero_sets_random_noise_0(self):
        mask = compile_deamon_wiring(_square_wiring(exc_noise=0.0, inh_noise=0.0))
        assert all(d["random_noise"] == 0.0 for d in mask)

    def test_noise_does_not_modify_exc_template_weights(self):
        """noise=0.67 must NOT call _apply_noise on the excitatory template."""
        mask_noisy = compile_deamon_wiring(_square_wiring(exc_noise=0.67))
        mask_clean = compile_deamon_wiring(_square_wiring())
        exc_noisy = next(d for d in mask_noisy if d["peso_dendrita"] > 0)
        exc_clean  = next(d for d in mask_clean  if d["peso_dendrita"] > 0)
        assert exc_noisy["pesos_sinapsis"] == exc_clean["pesos_sinapsis"]

    def test_noise_does_not_modify_inh_template_weights(self):
        """noise=0.67 must NOT call _apply_noise on the inhibitory template."""
        mask_noisy = compile_deamon_wiring(_square_wiring(inh_noise=0.67))
        mask_clean = compile_deamon_wiring(_square_wiring())
        inh_noisy = [d for d in mask_noisy if d["peso_dendrita"] < 0]
        inh_clean  = [d for d in mask_clean  if d["peso_dendrita"] < 0]
        for d_n, d_c in zip(inh_noisy, inh_clean):
            assert d_n["pesos_sinapsis"] == d_c["pesos_sinapsis"]

    def test_square_flower_petals_get_random_noise(self):
        wiring = {
            "shape": "square_flower",
            "excitatory": {"offset": 1, "weights": [1], "noise": 0.3},
            "inhibitory": {"offset": 5, "weights": [1], "multiplier": 4, "noise": 0.3},
        }
        mask = compile_deamon_wiring(wiring)
        inh = [d for d in mask if d["peso_dendrita"] < 0]
        assert len(inh) == 4
        assert all(d["random_noise"] == pytest.approx(0.3) for d in inh)

    def test_square_structure_one_exc_plus_sectors(self):
        mask = compile_deamon_wiring(_square_wiring(sectors=6))
        exc = [d for d in mask if d["peso_dendrita"] > 0]
        inh = [d for d in mask if d["peso_dendrita"] < 0]
        assert len(exc) == 1
        assert len(inh) == 6


# ── aplicar_mascara_2d ────────────────────────────────────────────────────────

class TestAplicarMascara2dNoise:
    """aplicar_mascara_2d uses random_noise to determine the weight scaling range."""

    def test_random_noise_zero_gives_exact_weights(self):
        """random_noise=0 → each synapse gets exactly its pesos_sinapsis value."""
        dendrite = {
            "peso_dendrita": 1.0,
            "offsets": [(1, 0), (-1, 0)],
            "pesos_sinapsis": [0.8, 0.6],
            "random_noise": 0.0,
        }
        random.seed(0)
        brain = _build_brain_with_mask([dendrite])
        n = brain.get_neurona("x10y10")
        weights = {s.neurona_entrante.id: s.peso for d in n.dendritas for s in d.sinapsis}
        assert weights["x11y10"] == pytest.approx(0.8)
        assert weights["x9y10"] == pytest.approx(0.6)

    def test_random_noise_0_67_weights_in_expected_range(self):
        """random_noise=0.67 → weight = base * uniform(0.33, 1.0)."""
        base = 0.9
        noise = 0.67
        dendrite = {
            "peso_dendrita": 1.0,
            "offsets": [(dx, 0) for dx in range(1, 11)],
            "pesos_sinapsis": [base] * 10,
            "random_noise": noise,
        }
        random.seed(42)
        brain = _build_brain_with_mask([dendrite])
        n = brain.get_neurona("x10y10")
        weights = [s.peso for s in n.dendritas[0].sinapsis]
        lo, hi = base * (1.0 - noise), base
        assert all(lo - 1e-9 <= w <= hi + 1e-9 for w in weights)

    def test_random_noise_0_67_mean_above_daemon_threshold(self):
        """With noise=0.67, mean weight > 0.622 — the analytical threshold for
        daemon formation under avg_vs_avg with exc_w=0.9, inh_w=0.68."""
        noise = 0.67
        n_neurons = 50
        dendrite = {
            "peso_dendrita": 1.0,
            "offsets": [(1, 0)],
            "pesos_sinapsis": [1.0],
            "random_noise": noise,
        }
        random.seed(7)
        brain = _build_brain_with_mask([dendrite], width=60, height=60)
        all_weights = [
            s.peso
            for x in range(5, 5 + n_neurons)
            for d in brain.get_neurona(f"x{x}y30").dendritas
            for s in d.sinapsis
        ]
        mean_w = sum(all_weights) / len(all_weights)
        assert mean_w > 0.622, f"mean={mean_w:.3f} below daemon threshold 0.622"

    def test_no_random_noise_field_uses_legacy_range(self):
        """Preset dendrites (no random_noise key) keep legacy uniform(0.2, 1.0) range."""
        dendrite = {
            "peso_dendrita": 1.0,
            "offsets": [(dx, 0) for dx in range(1, 21)],
            "pesos_sinapsis": [1.0] * 20,
            # intentionally no "random_noise" key
        }
        random.seed(0)
        brain = _build_brain_with_mask([dendrite])
        n = brain.get_neurona("x10y10")
        weights = [s.peso for s in n.dendritas[0].sinapsis]
        assert all(0.2 - 1e-9 <= w <= 1.0 + 1e-9 for w in weights)
        # at least some below 0.5 (would not happen with the 0.5-default formula)
        assert any(w < 0.5 for w in weights)

    def test_random_weights_false_ignores_random_noise(self):
        """random_weights=False → exact template weights even if random_noise is set."""
        dendrite = {
            "peso_dendrita": 1.0,
            "offsets": [(1, 0), (-1, 0)],
            "pesos_sinapsis": [0.8, 0.6],
            "random_noise": 0.67,
        }
        brain = _build_brain_with_mask([dendrite], random_weights=False)
        n = brain.get_neurona("x10y10")
        weights = {s.neurona_entrante.id: s.peso for d in n.dendritas for s in d.sinapsis}
        assert weights["x11y10"] == pytest.approx(0.8)
        assert weights["x9y10"] == pytest.approx(0.6)


# ── _compute_preview_grid ─────────────────────────────────────────────────────

class TestPreviewGridNoise:
    """_compute_preview_grid uses the same noise formula as the live network."""

    def test_random_noise_zero_gives_exact_preview(self):
        """With random_noise=0, preview weight equals pesos_sinapsis exactly."""
        mask = [{
            "peso_dendrita": 1.0,
            "offsets": [(1, 0)],
            "pesos_sinapsis": [0.75],
            "random_noise": 0.0,
        }]
        grid = _compute_preview_grid(mask, grid_width=20, grid_height=20, random_weights=True)
        cy, cx = 10, 10
        assert grid[cy][cx + 1] == pytest.approx(0.75)

    def test_random_noise_0_67_preview_in_range(self):
        """With random_noise=0.67, preview weight = base * uniform(0.33, 1.0)."""
        base = 0.9
        noise = 0.67
        mask = [{
            "peso_dendrita": 1.0,
            "offsets": [(dx, 0) for dx in range(1, 11)],
            "pesos_sinapsis": [base] * 10,
            "random_noise": noise,
        }]
        grid = _compute_preview_grid(mask, grid_width=30, grid_height=10, random_weights=True)
        cy, cx = 5, 15
        weights = [grid[cy][cx + dx] for dx in range(1, 11)]
        assert all(w is not None for w in weights)
        lo, hi = base * (1.0 - noise), base
        assert all(lo - 1e-9 <= w <= hi + 1e-9 for w in weights)

    def test_random_weights_false_overrides_random_noise_in_preview(self):
        """random_weights=False → exact weights in preview even with random_noise set."""
        mask = [{
            "peso_dendrita": 1.0,
            "offsets": [(1, 0)],
            "pesos_sinapsis": [0.75],
            "random_noise": 0.67,
        }]
        grid = _compute_preview_grid(mask, grid_width=20, grid_height=20, random_weights=False)
        cy, cx = 10, 10
        assert grid[cy][cx + 1] == pytest.approx(0.75)

    def test_preview_and_constructor_agree_on_noise_zero(self):
        """Both preview and constructor produce exact weights when random_noise=0."""
        mask = compile_deamon_wiring(_square_wiring(exc_noise=0.0, inh_noise=0.0))

        # preview: exact pesos_sinapsis (all 1.0 for weights=[1,1,1])
        grid = _compute_preview_grid(mask, grid_width=40, grid_height=40, random_weights=True)
        cy, cx = 20, 20
        # excitatory offsets are at distance 1..3 — any synapse cell is exactly 1.0
        cells = [
            grid[cy + dy][cx + dx]
            for dx in range(-3, 4)
            for dy in range(-3, 4)
            if grid[cy + dy][cx + dx] not in (None, 999.0)
        ]
        assert all(c == pytest.approx(1.0) or c == pytest.approx(-1.0) for c in cells)

        # constructor: also exact weights
        random.seed(99)
        brain = _build_brain_with_mask(mask, width=40, height=40)
        n = brain.get_neurona("x20y20")
        exc_d = next(d for d in n.dendritas if d.peso > 0)
        assert all(abs(s.peso - 1.0) < 1e-9 for s in exc_d.sinapsis)

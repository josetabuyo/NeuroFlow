"""REST API routes for NeuroFlow."""

from __future__ import annotations

from fastapi import APIRouter, Request

from core.masks import get_mask_info
from core.ascii_renderer import get_available_fonts
from db import save_config, get_latest, get_history

router = APIRouter(prefix="/api")

EXPERIMENTS = {
    "deamons_lab": {
        "id": "deamons_lab",
        "name": "Deamons Lab",
        "description": (
            "Configurable mask connectivity lab. "
            "Choose from multiple Mexican hat presets and adjust "
            "the excitation/inhibition balance. "
            "Supports hot reconnection without losing state."
        ),
        "masks": get_mask_info(),
        "balance_modes": [
            {"id": "none", "name": "No balance"},
            {"id": "weight", "name": "By weight"},
            {"id": "synapse_count", "name": "By synapse count"},
        ],
        "default_config": {
            "width": 50,
            "height": 50,
            "mask": "deamon_3_en_50",
            "balance": 0.0,
            "balance_mode": "none",
            "process_mode": "min_vs_max",
        },
    },
    "dynamic_som": {
        "id": "dynamic_som",
        "name": "Dynamic SOM",
        "description": (
            "Self-organizing map with visual input streams. "
            "Projects ASCII characters (with configurable noise) "
            "into an input region connected to a daemon tissue."
        ),
        "masks": get_mask_info(),
        "input_sources": [
            {"id": "ascii", "name": "ASCII Images"},
        ],
        "fonts": get_available_fonts(),
        "default_config": {
            "width": 50,
            "height": 50,
            "mask": "deamon_3_en_50",
            "input_text": "AB",
            "input_resolution": 20,
            "frames_per_char": 10,
            "input_dendrite_weight": 0.2,
            "deamon_exc_weight": 0.5,
            "deamon_inh_weight": -0.5,
            "background_white_noise": 0.05,
            "shift_noise": False,
            "noise_inter_char": True,
            "input_source": "ascii",
            "font": "press_start_2p",
            "font_size": 10,
            "learning": True,
            "learning_rate": 0.01,
            "spike_adaptation": False,
            "max_active_steps": 5,
            "refractory_steps": 5,
            "process_mode": "min_vs_max",
            "tension_function": {},
        },
        "config_presets": [
            # ── Tier 0: Daemon puro (sin input, sin aprendizaje) ──
            {
                "id": "isolated_daemon",
                "name": "Isolated Daemon",
                "description": (
                    "Pure daemon formation without input or learning. "
                    "Replicates Deamons Lab inside the SOM topology. "
                    "Look for stable Mexican hat circles in the tissue."
                ),
                "config": {
                    "description": (
                        "Daemon-only baseline: no input, no learning. Verifies that the "
                        "Mexican hat topology forms stable white circles (daemons). "
                        "Nivel 0 of the SOM validation ladder."
                    ),
                    "width": 50,
                    "height": 50,
                    "mask": "deamon_3_en_50",
                    "input_text": "",
                    "deamon_exc_weight": 1,
                    "deamon_inh_weight": -1,
                    "learning": False,
                    "process_mode": "min_vs_max",
                    "tension_function": {},
                },
            },
            {
                "id": "daemon_adaptation",
                "name": "Daemon + Adaptation",
                "description": (
                    "Daemons with spike frequency adaptation (ON/OFF cycling). "
                    "Shows the 'lava lamp' effect: daemons migrate and die. "
                    "Key finding: dynamism without organization (Problem 4)."
                ),
                "config": {
                    "description": (
                        "Spike frequency adaptation creates ON/OFF cycling. Daemons birth, "
                        "migrate, and die — visually beautiful 'lava lamp' but does NOT produce "
                        "topological organization. Dynamism is necessary but not sufficient."
                    ),
                    "width": 50,
                    "height": 50,
                    "mask": "deamon_3_en_50",
                    "input_text": "",
                    "deamon_exc_weight": 0.5,
                    "deamon_inh_weight": -0.5,
                    "learning": False,
                    "spike_adaptation": True,
                    "max_active_steps": 5,
                    "refractory_steps": 5,
                    "process_mode": "min_vs_max",
                    "tension_function": {},
                },
            },
            # ── Tier 1: Patrones simples — investigación del techo plano ──
            {
                "id": "halves_raw",
                "name": "Halves Raw",
                "description": (
                    "Simplest binary patterns with raw tension. "
                    "Demonstrates the flat ceiling problem: all neurons "
                    "have similar tension, preventing differentiation."
                ),
                "config": {
                    "description": (
                        "HALF_TOP/HALF_BOT with no tension sharpening. Shows the flat ceiling: "
                        "tension is similar across all neurons (~0.3-0.5), so learning affects "
                        "everyone equally. No winner, no organization. Baseline for comparison."
                    ),
                    "width": 50,
                    "height": 50,
                    "mask": "deamon_3_en_50",
                    "input_text": "HALF_TOP,HALF_BOT",
                    "input_resolution": 20,
                    "frames_per_char": 20,
                    "input_dendrite_weight": 0.2,
                    "deamon_exc_weight": 0.5,
                    "deamon_inh_weight": -0.5,
                    "background_white_noise": 0,
                    "learning_rate": 0.01,
                    "process_mode": "min_vs_max",
                    "tension_function": {},
                },
            },
            {
                "id": "halves_pow2",
                "name": "Halves Sharp pow 2",
                "description": (
                    "Light power-law tension sharpening (exponent 2). "
                    "Tension 0.5 drops to 0.25, 0.9 stays at 0.81. "
                    "First attempt at creating tension peaks."
                ),
                "config": {
                    "description": (
                        "Power-law sharpening with exponent 2 applied to tension before "
                        "learning. sign(t)*|t|^2 compresses mid-range tensions while preserving "
                        "peaks. Compare inspect weights with Halves Raw to see the difference."
                    ),
                    "width": 50,
                    "height": 50,
                    "mask": "deamon_3_en_50",
                    "input_text": "HALF_TOP,HALF_BOT",
                    "input_resolution": 20,
                    "frames_per_char": 20,
                    "input_dendrite_weight": 0.2,
                    "deamon_exc_weight": 0.5,
                    "deamon_inh_weight": -0.5,
                    "background_white_noise": 0,
                    "learning_rate": 0.01,
                    "process_mode": "min_vs_max",
                    "tension_function": {"pow": 2.0},
                },
            },
            {
                "id": "halves_pow3",
                "name": "Halves Sharp pow 3",
                "description": (
                    "Strong sharpening (exponent 3). Tension 0.3 becomes 0.027, "
                    "0.9 stays at 0.729. Highly selective learning — "
                    "only the most tensioned neurons learn significantly."
                ),
                "config": {
                    "description": (
                        "Exponent 3 creates strong selectivity: neurons with tension 0.3 "
                        "effectively get lr*0.027 while 0.9 gets lr*0.729 — a 27x ratio. "
                        "This should concentrate learning on winner neurons if any exist."
                    ),
                    "width": 50,
                    "height": 50,
                    "mask": "deamon_3_en_50",
                    "input_text": "HALF_TOP,HALF_BOT",
                    "input_resolution": 20,
                    "frames_per_char": 20,
                    "input_dendrite_weight": 0.2,
                    "deamon_exc_weight": 0.5,
                    "deamon_inh_weight": -0.5,
                    "background_white_noise": 0,
                    "learning_rate": 0.01,
                    "process_mode": "min_vs_max",
                    "tension_function": {"pow": 3.0},
                },
            },
            {
                "id": "halves_pow5",
                "name": "Halves Sharp pow 5",
                "description": (
                    "Extreme sharpening (exponent 5). Near-binary selectivity: "
                    "only neurons with very high tension learn. "
                    "Useful to find the optimal selectivity threshold."
                ),
                "config": {
                    "description": (
                        "Exponent 5 is near winner-take-all: tension 0.5 becomes 0.031, "
                        "0.9 becomes 0.59. Almost only the peak neurons learn. "
                        "If this works but pow 2 doesn't, the ceiling needs aggressive breaking."
                    ),
                    "width": 50,
                    "height": 50,
                    "mask": "deamon_3_en_50",
                    "input_text": "HALF_TOP,HALF_BOT",
                    "input_resolution": 20,
                    "frames_per_char": 20,
                    "input_dendrite_weight": 0.2,
                    "deamon_exc_weight": 0.5,
                    "deamon_inh_weight": -0.5,
                    "background_white_noise": 0,
                    "learning_rate": 0.01,
                    "process_mode": "min_vs_max",
                    "tension_function": {"pow": 5.0},
                },
            },
            # ── Tier 2: Diversidad de patrones ──
            {
                "id": "bars_sharp",
                "name": "Bars Sharp pow 3",
                "description": (
                    "Horizontal and vertical bar patterns with pow-3 sharpening. "
                    "Tests organization with richer spatial structure. "
                    "Run after validating halves work."
                ),
                "config": {
                    "description": (
                        "BARS_H (3 horizontal bars) and BARS_V (3 vertical bars) with pow-3 "
                        "sharpening. More complex spatial patterns than halves — tests whether "
                        "the tissue can develop different receptive fields for each orientation."
                    ),
                    "width": 50,
                    "height": 50,
                    "mask": "deamon_3_en_50",
                    "input_text": "BARS_H,BARS_V",
                    "input_resolution": 20,
                    "frames_per_char": 20,
                    "input_dendrite_weight": 0.2,
                    "deamon_exc_weight": 0.5,
                    "deamon_inh_weight": -0.5,
                    "background_white_noise": 0,
                    "learning_rate": 0.01,
                    "process_mode": "min_vs_max",
                    "tension_function": {"pow": 3.0},
                },
            },
            {
                "id": "dots_sharp",
                "name": "Corner Dots pow 3",
                "description": (
                    "Dots in opposite corners with pow-3 sharpening. "
                    "Sparse localized patterns to test spatial selectivity "
                    "for small, well-separated features."
                ),
                "config": {
                    "description": (
                        "DOT_TL (5x5 top-left) and DOT_BR (5x5 bottom-right) with pow-3 "
                        "sharpening. Maximally separated sparse patterns — if the tissue can't "
                        "differentiate these, the problem is deeper than sharpening."
                    ),
                    "width": 50,
                    "height": 50,
                    "mask": "deamon_3_en_50",
                    "input_text": "DOT_TL,DOT_BR",
                    "input_resolution": 20,
                    "frames_per_char": 20,
                    "input_dendrite_weight": 0.2,
                    "deamon_exc_weight": 0.5,
                    "deamon_inh_weight": -0.5,
                    "background_white_noise": 0,
                    "learning_rate": 0.01,
                    "process_mode": "min_vs_max",
                    "tension_function": {"pow": 3.0},
                },
            },
            # ── Tier 3: ASCII — el desafío original ──
            {
                "id": "ascii_baseline",
                "name": "ASCII Baseline",
                "description": (
                    "Original Dynamic SOM config: ASCII A/B with white noise. "
                    "No sharpening. Historical baseline — known to NOT converge "
                    "due to flat tension ceiling."
                ),
                "config": {
                    "description": (
                        "The original experiment setup from Stage 2. Characters A and B "
                        "rendered with press_start_2p font, white noise between chars. "
                        "Known result: no convergence, weights stay noisy (Problem 3)."
                    ),
                    "width": 50,
                    "height": 50,
                    "mask": "deamon_3_en_50",
                    "input_text": "AB",
                    "input_resolution": 20,
                    "frames_per_char": 10,
                    "input_dendrite_weight": 0.2,
                    "deamon_exc_weight": 0.5,
                    "deamon_inh_weight": -0.5,
                    "learning_rate": 0.01,
                    "font": "press_start_2p",
                    "font_size": 10,
                    "process_mode": "min_vs_max",
                    "tension_function": {},
                },
            },
            {
                "id": "ascii_sharp",
                "name": "ASCII Sharp pow 3",
                "description": (
                    "ASCII A/B with pow-3 tension sharpening. "
                    "Best available config for the original challenge. "
                    "Compare with ASCII Baseline to measure improvement."
                ),
                "config": {
                    "description": (
                        "ASCII characters A/B with pow-3 sharpening applied. Combines real "
                        "character shapes with selective learning. Inspect neuron weights after "
                        "~500 steps and compare with ASCII Baseline to see differentiation."
                    ),
                    "width": 50,
                    "height": 50,
                    "mask": "deamon_3_en_50",
                    "input_text": "AB",
                    "input_resolution": 20,
                    "frames_per_char": 10,
                    "input_dendrite_weight": 0.2,
                    "deamon_exc_weight": 0.5,
                    "deamon_inh_weight": -0.5,
                    "learning_rate": 0.01,
                    "font": "press_start_2p",
                    "font_size": 10,
                    "process_mode": "min_vs_max",
                    "tension_function": {"pow": 3.0},
                },
            },
            # ── Tier 4: Experimentos compuestos ──
            {
                "id": "halves_adaptation",
                "name": "Halves + Adaptation",
                "description": (
                    "HALF_TOP/HALF_BOT with spike adaptation AND pow-3 sharpening. "
                    "Tests whether the lava-lamp dynamism helps or hurts "
                    "when combined with tension sharpening."
                ),
                "config": {
                    "description": (
                        "Combines spike frequency adaptation (ON/OFF cycling) with pow-3 "
                        "tension sharpening. The adaptation forces daemon migration while "
                        "sharpening directs learning. Does movement help exploration?"
                    ),
                    "width": 50,
                    "height": 50,
                    "mask": "deamon_3_en_50",
                    "input_text": "HALF_TOP,HALF_BOT",
                    "input_resolution": 20,
                    "frames_per_char": 20,
                    "input_dendrite_weight": 0.2,
                    "deamon_exc_weight": 0.5,
                    "deamon_inh_weight": -0.5,
                    "background_white_noise": 0,
                    "learning_rate": 0.01,
                    "spike_adaptation": True,
                    "max_active_steps": 5,
                    "refractory_steps": 5,
                    "process_mode": "min_vs_max",
                    "tension_function": {"pow": 3.0},
                },
            },
            {
                "id": "halves_sum_mode",
                "name": "Halves Sum Mode",
                "description": (
                    "Halves with sum dendrite integration instead of min_vs_max, "
                    "plus pow-3 sharpening. Tests alternative dendrite combination. "
                    "Sum tends toward saturation — compare stability."
                ),
                "config": {
                    "description": (
                        "Uses process_mode 'sum' (all dendrites summed and clamped) instead of "
                        "min_vs_max (best excitatory vs best inhibitory). Sum mode gives wider "
                        "tension range but risks saturation. Compare daemon stability."
                    ),
                    "width": 50,
                    "height": 50,
                    "mask": "deamon_3_en_50",
                    "input_text": "HALF_TOP,HALF_BOT",
                    "input_resolution": 20,
                    "frames_per_char": 20,
                    "input_dendrite_weight": 0.2,
                    "deamon_exc_weight": 0.5,
                    "deamon_inh_weight": -0.5,
                    "background_white_noise": 0,
                    "learning_rate": 0.01,
                    "process_mode": "sum",
                    "tension_function": {"pow": 3.0},
                },
            },
        ],
    },
}


@router.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@router.get("/experiments")
async def list_experiments() -> list[dict]:
    """List of available experiments."""
    return list(EXPERIMENTS.values())


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict:
    """Config for a specific experiment."""
    if experiment_id not in EXPERIMENTS:
        return {"error": f"Experiment '{experiment_id}' not found"}
    return EXPERIMENTS[experiment_id]


@router.post("/experiments/{experiment_id}/config")
async def save_experiment_config(experiment_id: str, request: Request) -> dict:
    """Persist a config snapshot for an experiment."""
    config = await request.json()
    sid = save_config(experiment_id, config)
    return {"id": sid}


@router.get("/experiments/{experiment_id}/config/latest")
def get_latest_config(experiment_id: str) -> dict:
    """Return the most recently saved config for an experiment."""
    config = get_latest(experiment_id)
    return {"config": config}


@router.get("/experiments/{experiment_id}/config/history")
def get_config_history(experiment_id: str) -> dict:
    """All executed configs for an experiment, oldest first."""
    return {"history": get_history(experiment_id)}

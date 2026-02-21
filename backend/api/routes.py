"""REST API routes for NeuroFlow."""

from __future__ import annotations

from fastapi import APIRouter

from core.masks import get_mask_info

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
        },
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

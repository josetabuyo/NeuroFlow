"""REST API routes for NeuroFlow."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from core.masks import get_mask_info, preview_deamon_wiring
from core.ascii_renderer import get_available_fonts
from db import (
    create_experiment,
    delete_experiment,
    get_experiment,
    get_runs,
    list_experiments,
    save_run,
    update_experiment,
)

router = APIRouter(prefix="/api")


# ── Endpoints ──

@router.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.2.0"}


@router.get("/metadata")
async def get_metadata() -> dict:
    """Return available masks, fonts, process modes, and input sources."""
    return {
        "masks": get_mask_info(),
        "fonts": get_available_fonts(),
        "process_modes": [
            {"id": "min_vs_max", "name": "Min vs Max", "description": "Best excitatory vs best inhibitory"},
            {"id": "avg_vs_avg", "name": "Avg vs Avg", "description": "Average excitatory vs average inhibitory"},
            {"id": "avg_vs_avg_normalized", "name": "Avg vs Avg (Norm)", "description": "Ratio exc/inh — only balance matters, not absolute scale"},
            {"id": "sum", "name": "Sum", "description": "All dendrites summed and clamped"},
            {"id": "group_avg", "name": "Group Avg", "description": "Separate averages for local-exc, local-inh, cross-exc, cross-inh"},
        ],
        "input_sources": [
            {"id": "ascii", "name": "ASCII Images"},
        ],
    }


@router.post("/preview-wiring")
async def preview_wiring(request: Request) -> dict:
    """Compute preview_grid and mask_stats for an inline deamon wiring definition."""
    wiring = await request.json()
    return preview_deamon_wiring(wiring)


# ── Experiments (ABM) ──

@router.get("/experiments")
async def api_list_experiments() -> list[dict]:
    """List all experiments (lightweight — no config payload)."""
    return list_experiments()


@router.post("/experiments")
async def api_create_experiment(request: Request) -> dict:
    """Create a new experiment."""
    body = await request.json()
    name = body.get("name") or "Untitled"
    config = body.get("config", {})
    return create_experiment(name, config)


@router.get("/experiments/{experiment_id}")
async def api_get_experiment(experiment_id: int) -> dict:
    """Get a single experiment, including its current config."""
    experiment = get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return experiment


@router.patch("/experiments/{experiment_id}")
async def api_update_experiment(experiment_id: int, request: Request) -> dict:
    """Update an experiment's name, config, and/or position."""
    body = await request.json()
    experiment = update_experiment(
        experiment_id,
        name=body.get("name"),
        config=body.get("config"),
        position=body.get("position"),
    )
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return experiment


@router.delete("/experiments/{experiment_id}")
async def api_delete_experiment(experiment_id: int) -> dict:
    """Delete an experiment and its run history."""
    deleted = delete_experiment(experiment_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return {"ok": True}


@router.get("/experiments/{experiment_id}/runs")
async def api_get_runs(experiment_id: int) -> dict:
    """All executed configs for this experiment, oldest first."""
    return {"history": get_runs(experiment_id)}


@router.post("/experiments/{experiment_id}/runs")
async def api_save_run(experiment_id: int, request: Request) -> dict:
    """Persist a run snapshot for this experiment."""
    config = await request.json()
    run_id = save_run(experiment_id, config)
    return {"id": run_id}

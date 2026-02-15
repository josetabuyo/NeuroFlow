"""REST API routes for NeuroFlow."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api")

EXPERIMENTS = {
    "von_neumann": {
        "id": "von_neumann",
        "name": "Autómata Elemental (Von Neumann)",
        "description": (
            "Autómata celular elemental 1D (reglas de Wolfram) implementado "
            "con sinapsis, dendritas y neuronas. Cada fila es una generación "
            "del autómata, propagándose de abajo hacia arriba."
        ),
        "rules": [111, 30, 90, 110],
        "default_config": {
            "width": 50,
            "height": 50,
            "rule": 111,
        },
    }
}


@router.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@router.get("/experiments")
async def list_experiments() -> list[dict]:
    """Lista de experimentos disponibles."""
    return list(EXPERIMENTS.values())


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict:
    """Config de un experimento específico."""
    if experiment_id not in EXPERIMENTS:
        return {"error": f"Experiment '{experiment_id}' not found"}
    return EXPERIMENTS[experiment_id]

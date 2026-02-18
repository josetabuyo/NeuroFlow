"""REST API routes for NeuroFlow."""

from __future__ import annotations

from fastapi import APIRouter

from core.masks import get_mask_info

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
    },
    "kohonen": {
        "id": "kohonen",
        "name": "Kohonen (Competencia Lateral 2D)",
        "description": (
            "Mapa autoorganizado de Kohonen con excitación local e inhibición "
            "lateral. Perfil 'Mexican hat': las neuronas excitan a sus vecinas "
            "cercanas e inhiben a las lejanas, formando clusters que compiten."
        ),
        "default_config": {
            "width": 30,
            "height": 30,
        },
    },
    "kohonen_balanced": {
        "id": "kohonen_balanced",
        "name": "Kohonen Balanceado",
        "description": (
            "Kohonen con balance configurable del Fuzzy OR. Con balance=0 "
            "funciona idéntico al Kohonen simple. Valores positivos reducen "
            "inhibición (sesgo excitatorio), negativos reducen excitación "
            "(sesgo inhibitorio)."
        ),
        "default_config": {
            "width": 30,
            "height": 30,
            "balance": 0.0,
        },
    },
    "kohonen_lab": {
        "id": "kohonen_lab",
        "name": "Kohonen Lab",
        "description": (
            "Laboratorio de conexionados Kohonen. Elige entre múltiples presets "
            "de sombrero mexicano y ajusta el balance excitación/inhibición. "
            "Soporta reconexión en caliente sin perder estado."
        ),
        "masks": get_mask_info(),
        "balance_modes": [
            {"id": "none", "name": "Sin balance"},
            {"id": "weight", "name": "Por peso"},
            {"id": "synapse_count", "name": "Por cantidad de sinapsis"},
        ],
        "default_config": {
            "width": 30,
            "height": 30,
            "mask": "simple",
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
    """Lista de experimentos disponibles."""
    return list(EXPERIMENTS.values())


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict:
    """Config de un experimento específico."""
    if experiment_id not in EXPERIMENTS:
        return {"error": f"Experiment '{experiment_id}' not found"}
    return EXPERIMENTS[experiment_id]

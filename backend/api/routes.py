"""REST API routes for NeuroFlow."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request

from core.masks import get_mask_info
from core.ascii_renderer import get_available_fonts
from db import save_config, get_latest, get_history

router = APIRouter(prefix="/api")

# ── Config template loader ──

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def _load_templates() -> list[dict]:
    """Load all JSON config templates from the configs/ directory."""
    templates: list[dict] = []
    if not CONFIGS_DIR.is_dir():
        return templates
    for fp in sorted(CONFIGS_DIR.glob("*.json")):
        try:
            data = json.loads(fp.read_text())
            templates.append({
                "id": fp.stem,
                "name": data.get("name", fp.stem),
                "description": data.get("description", ""),
                "config": data.get("config", {}),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return templates


TEMPLATES: list[dict] = _load_templates()


# ── Endpoints ──

@router.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.2.0"}


@router.get("/templates")
async def list_templates() -> list[dict]:
    """List all available config templates."""
    return TEMPLATES


@router.post("/templates/refresh")
async def refresh_templates() -> list[dict]:
    """Re-read all JSON files from configs/ and update the in-memory TEMPLATES list."""
    global TEMPLATES
    TEMPLATES = _load_templates()
    return TEMPLATES


@router.get("/templates/{template_id}")
async def get_template(template_id: str) -> dict:
    """Get a specific config template."""
    for t in TEMPLATES:
        if t["id"] == template_id:
            return t
    return {"error": f"Template '{template_id}' not found"}


@router.get("/metadata")
async def get_metadata() -> dict:
    """Return available masks, fonts, process modes, and input sources."""
    return {
        "masks": get_mask_info(),
        "fonts": get_available_fonts(),
        "process_modes": [
            {"id": "min_vs_max", "name": "Min vs Max", "description": "Best excitatory vs best inhibitory"},
            {"id": "avg_vs_avg", "name": "Avg vs Avg", "description": "Average excitatory vs average inhibitory"},
            {"id": "sum", "name": "Sum", "description": "All dendrites summed and clamped"},
        ],
        "input_sources": [
            {"id": "ascii", "name": "ASCII Images"},
        ],
    }


@router.post("/templates/{template_id}/config")
async def save_template_config(
    template_id: str, request: Request, preset: str = "_default",
) -> dict:
    """Persist a config snapshot for a template+preset."""
    config = await request.json()
    sid = save_config(template_id, preset, config)
    return {"id": sid}


@router.get("/templates/{template_id}/config/latest")
def get_latest_config(template_id: str, preset: str = "_default") -> dict:
    """Return the most recently saved config for a template+preset."""
    config = get_latest(template_id, preset)
    return {"config": config}


@router.get("/templates/{template_id}/config/history")
def get_config_history(template_id: str, preset: str = "_default") -> dict:
    """All executed configs for a template+preset, oldest first."""
    return {"history": get_history(template_id, preset)}

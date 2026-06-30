"""Unified NeuroFlow experiment — config-driven, region-agnostic.

The network is fully described by the canonical config (regions + connections):

  - regions[].wiring  → Neurona cells with internal connections (daemon/mask),
                         processed by BrainTensor every step.
  - regions[].source  → NeuronaEntrada cells, values set externally each step:
        "ascii"      → synthetic / rendered text patterns
        "label"      → ground-truth class signal
        "error_diff"      → |target - output| diff (nociceptor, legacy)
        "label_mismatch"  → 1 if neuron's label ≠ current input char, else 0
  - connections[]     → cross-region dendrites, with weight / density / learning.rate.

A region may declare wiring AND/OR source (orthogonal). Nothing here is keyed on a
specific region name — everything flows from the config. Legacy flat configs are
auto-converted via _to_canonical().
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from core.constructor import Constructor
from core.constructor_tensor import ConstructorTensor
from core.neurona import Neurona, NeuronaEntrada
from core.brain import Brain
from core.region import Region
from core.sinapsis import Sinapsis
from core.dendrita import Dendrita
from core.masks import get_mask, get_mask_type, get_random_weights, compile_deamon_wiring
from core.nerve import place_nerve_circles, circle_cells_with_weights, NerveCircle
from core.ascii_renderer import render_char, apply_white_noise, apply_shift_noise
from .base import Experimento

logger = logging.getLogger(__name__)

_STABILITY_WINDOW = 20
_DAEMON_THRESHOLD = 0.5
_MIN_DAEMON_SIZE = 3

_SOURCE_TYPES_NON_INPUT = {"label", "error_diff", "label_mismatch"}
_NOCICEPTOR_SOURCE_TYPES = {"label_mismatch", "error_diff"}
_SYNTHETIC_PATTERNS = {"HALF_TOP", "HALF_BOT", "BARS_H", "BARS_V", "DOT_TL", "DOT_BR"}


@dataclass
class RegionState:
    """Everything the runtime needs to know about one region.

    Spans [start, end) index into the global neuron tensor in declaration order.
    """

    id: str
    start: int
    end: int
    width: int
    height: int
    is_entrada: bool
    source_type: str | None = None
    source_cfg: dict[str, Any] = field(default_factory=dict)
    wiring_cfg: dict[str, Any] = field(default_factory=dict)
    umbral: float = 0.0
    process_mode: str = "min_vs_max"
    tension_fns: list[tuple[str, float]] = field(default_factory=list)
    soft_activation: bool = False
    neuron_labels: list[list[str]] | None = None

    handler_method: str | None = None

    # Runtime state (source regions)
    char_index: int = 0
    frame_in_char: int = 0
    in_gap: bool = False
    current_frame: np.ndarray | None = None
    char_images: dict[str, np.ndarray] = field(default_factory=dict)
    # draw source: clean painted values, kept separate from brain_tensor
    draw_base: torch.Tensor | None = None

    @property
    def n(self) -> int:
        return self.end - self.start

    @property
    def has_wiring(self) -> bool:
        return bool(self.wiring_cfg)

    @property
    def is_ascii_input(self) -> bool:
        return self.source_type == "ascii"


def _detect_daemons(
    values: torch.Tensor,
    width: int,
    height: int,
    threshold: float,
    min_size: int = _MIN_DAEMON_SIZE,
) -> tuple[int, set[int], set[int], list[int]]:
    """Detect daemons as connected components of active neurons (8-connectivity)."""
    n = width * height
    active = (values[:n] > threshold).tolist()
    visited = [False] * n
    daemon_indices: set[int] = set()
    noise_indices: set[int] = set()
    sizes: list[int] = []

    for idx in range(n):
        if active[idx] and not visited[idx]:
            queue = [idx]
            visited[idx] = True
            cluster: list[int] = []
            head = 0
            while head < len(queue):
                cidx = queue[head]
                head += 1
                cluster.append(cidx)
                cx, cy = cidx % width, cidx // width
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            nidx = ny * width + nx
                            if active[nidx] and not visited[nidx]:
                                visited[nidx] = True
                                queue.append(nidx)
            if len(cluster) >= min_size:
                daemon_indices.update(cluster)
                sizes.append(len(cluster))
            else:
                noise_indices.update(cluster)

    return len(sizes), daemon_indices, noise_indices, sizes


def _migrate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate old-format connections and region wirings to the new canonical format.

    Old → new:
    - {"type":"full","weight":w,...} → {"full":{"weight":w,...}}
    - {"type":"deamon","shape":s,"dendrite_exc_weight":w,...} → {"deamon":{"shape":s,"excitatory":{"weight":w,...},...}}
    - region.wiring {"mask":m,"dendrite_exc_weight":w} → {"deamon":{"mask":m,"excitatory":{"weight":w}}}
    """
    import copy
    config = copy.deepcopy(config)
    new_conns: list[dict[str, Any]] = []
    for conn in config.get("connections", []):
        ctype = conn.get("type")
        if "on" in conn and "deamon" not in conn and "nerve" not in conn:
            # Old-style intra-region connection → deamon key
            deamon: dict[str, Any] = {}
            for k in ("mask", "shape", "centroid", "fixed"):
                if k in conn:
                    deamon[k] = conn[k]
            exc_w = conn.get("dendrite_exc_weight")
            inh_w = conn.get("dendrite_inh_weight")
            if "excitatory" in conn:
                exc = dict(conn["excitatory"])
                if exc_w is not None:
                    exc = {"weight": exc_w, **{k: v for k, v in exc.items() if k != "weight"}}
                deamon["excitatory"] = exc
            elif exc_w is not None:
                deamon["excitatory"] = {"weight": exc_w}
            if "inhibitory" in conn:
                inh = dict(conn["inhibitory"])
                if inh_w is not None:
                    inh = {"weight": inh_w, **{k: v for k, v in inh.items() if k != "weight"}}
                deamon["inhibitory"] = inh
            elif inh_w is not None:
                deamon["inhibitory"] = {"weight": inh_w}
            if "learning_rate" in conn:
                deamon["learning"] = {"rate": conn["learning_rate"]}
            new_conns.append({"on": conn["on"], "deamon": deamon})
        elif ctype == "full" or ("from" in conn and "to" in conn and "full" not in conn and ctype != "portion"):
            # Old-style inter-region full connection → full key
            full: dict[str, Any] = {}
            if "weight" in conn:
                full["weight"] = conn["weight"]
            if "density" in conn:
                full["density"] = conn["density"]
            if "learning" in conn:
                full["learning"] = conn["learning"]
            elif "learning_rate" in conn:
                full["learning"] = {"rate": conn["learning_rate"]}
            new_conns.append({"from": conn["from"], "to": conn["to"], "full": full})
        else:
            new_conns.append(conn)
    if config.get("connections") is not None:
        config["connections"] = new_conns

    for region in config.get("regions", []):
        w = region.get("wiring")
        if not w or "deamon" in w:
            continue
        top_mask = w.get("mask", "")
        if top_mask and get_mask_type(top_mask) == "wolfram":
            continue
        deamon = {}
        if top_mask:
            deamon["mask"] = top_mask
        exc_w = w.get("dendrite_exc_weight")
        inh_w = w.get("dendrite_inh_weight")
        if exc_w is not None:
            deamon["excitatory"] = {"weight": exc_w}
        if inh_w is not None:
            deamon["inhibitory"] = {"weight": inh_w}
        new_w: dict[str, Any] = {"deamon": deamon}
        for k in ("process_mode", "learning_rate", "tension"):
            if k in w:
                new_w[k] = w[k]
        region["wiring"] = new_w

    return config


def _inject_wiring_from_connections(config: dict[str, Any]) -> dict[str, Any]:
    """Absorb on-connections into region wiring for internal use. Returns a deep copy."""
    import copy
    config = copy.deepcopy(config)
    connections = config.get("connections", [])
    on_conns = [c for c in connections if "on" in c and "nerve" not in c]
    if not on_conns:
        return config
    region_map = {r["id"]: r for r in config.get("regions", [])}
    for conn in on_conns:
        rid = conn["on"]
        if rid not in region_map:
            continue
        r = region_map[rid]
        if "wiring" in r:
            continue  # region wiring already present — on-connection skipped
        wiring: dict[str, Any] = {}
        pm = r.get("process_mode") or conn.get("process_mode")
        if pm:
            wiring["process_mode"] = pm
        if "deamon" in conn:
            deamon_cfg = conn["deamon"]
            wiring["deamon"] = deamon_cfg
            lr = deamon_cfg.get("learning", {}).get("rate")
            if lr is not None:
                wiring["learning_rate"] = lr
        r["wiring"] = wiring
    config["connections"] = [c for c in connections if "on" not in c or "nerve" in c]
    return config


def _to_canonical(config: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy flat config to the canonical regions/connections format.

    Already canonical (has 'regions') → returned as-is.
    Legacy (has top-level 'grid') → single tissue region + optional input region.
    """
    if "regions" in config:
        return config

    config = {**config}

    if "grid" not in config:
        logger.warning("config missing 'grid', defaulting to {width: 50, height: 50}")
        config["grid"] = {"width": 50, "height": 50}
    if "wiring" not in config:
        logger.warning("config missing 'wiring', defaulting to mask='deamon_3_en_50', process_mode='min_vs_max'")
        config["wiring"] = {"mask": "deamon_3_en_50", "process_mode": "min_vs_max"}

    grid = {**config["grid"]}
    grid.setdefault("width", 50)
    grid.setdefault("height", 50)

    wiring: dict[str, Any] = {**config["wiring"]}
    if "mask" not in wiring and "deamon" not in wiring:
        logger.warning("wiring.mask missing, defaulting to 'deamon_3_en_50'")
        wiring["mask"] = "deamon_3_en_50"
    wiring.setdefault("process_mode", "min_vs_max")

    input_cfg: dict[str, Any] | None = config.get("input")
    learning_cfg: dict[str, Any] | None = config.get("learning")
    noise_cfg = config.get("noise")
    spiking_cfg = config.get("spiking")
    daemon_cfg = config.get("daemon")

    if learning_cfg is not None:
        base_rate = float(learning_cfg.get("rate", 1.0))
        lr_exc = float(learning_cfg.get("lr_exc", 1.0))
        lr_inh = float(learning_cfg.get("lr_inh", 1.0))
        effective_wiring = base_rate * max(lr_exc, lr_inh)
        if effective_wiring > 0:
            wiring["learning_rate"] = effective_wiring

    tissue: dict[str, Any] = {"id": "tissue", "grid": grid, "wiring": wiring}
    if spiking_cfg:
        tissue["spiking"] = spiking_cfg
    if daemon_cfg:
        tissue["daemon"] = daemon_cfg

    regions: list[dict[str, Any]] = [tissue]
    connections: list[dict[str, Any]] = []

    if input_cfg is not None:
        resolution = int(input_cfg.get("resolution", 20))
        source: dict[str, Any] = {
            "type": "ascii",
            "text": input_cfg.get("text", "") or "",
            "frames_per_char": int(input_cfg.get("frames_per_char", 10)),
        }
        for key in ("font", "font_size"):
            if key in input_cfg:
                source[key] = input_cfg[key]
        if noise_cfg:
            source["noise"] = noise_cfg

        regions.insert(0, {
            "id": "input",
            "grid": {"width": resolution, "height": resolution},
            "source": source,
        })

        portion = input_cfg.get("portion")
        if portion is not None:
            conn: dict[str, Any] = {
                "from": "input",
                "to": "tissue",
                "type": "portion",
                "portion": list(portion),
                "weight": float(input_cfg.get("dendrite_input_weight", 0.2)),
            }
        else:
            full_cfg: dict[str, Any] = {
                "weight": float(input_cfg.get("dendrite_input_weight", 0.2)),
            }
            if "density" in input_cfg:
                full_cfg["density"] = float(input_cfg["density"])
            if learning_cfg is not None:
                base_rate = float(learning_cfg.get("rate", 1.0))
                lr_input = float(learning_cfg.get("lr_input", 1.0))
                effective_input = base_rate * lr_input
                if effective_input > 0:
                    full_cfg["learning"] = {"rate": effective_input}
            conn = {"from": "input", "to": "tissue", "full": full_cfg}

        connections.append(conn)

    canonical: dict[str, Any] = {}
    for key in ("name", "description"):
        if key in config:
            canonical[key] = config[key]
    canonical["regions"] = regions
    if connections:
        canonical["connections"] = connections

    return canonical


def _apply_mask_to_neuron_grid(
    neurons: list[list["Neurona"]],
    mask: list[dict],
    random_weights: bool,
) -> None:
    """Apply a wiring mask spatially to a 2D grid of neurons (toroidal wrap-around)."""
    height = len(neurons)
    width = len(neurons[0]) if neurons else 0
    if not width or not height:
        return
    for y in range(height):
        for x in range(width):
            dest = neurons[y][x]
            for dend_def in mask:
                peso_d: float = dend_def["peso_dendrita"]
                offsets: list[tuple[int, int]] = dend_def["offsets"]
                pesos_s: list[float] | None = dend_def.get("pesos_sinapsis")
                noise_amp = dend_def.get("random_noise")
                sinapsis_list: list[Sinapsis] = []
                for i, (dx, dy) in enumerate(offsets):
                    nx = (x + dx) % width
                    ny = (y + dy) % height
                    src = neurons[ny][nx]
                    base = pesos_s[i] if pesos_s is not None else 1.0
                    if not random_weights:
                        peso = base
                    elif noise_amp is not None:
                        scale = random.uniform(1.0 - noise_amp, 1.0) if noise_amp > 0 else 1.0
                        peso = base * scale
                    else:
                        peso = base * random.uniform(0.2, 1.0)
                    sinapsis_list.append(Sinapsis(neurona_entrante=src, peso=peso))
                if sinapsis_list:
                    dest.dendritas.append(Dendrita(sinapsis=sinapsis_list, peso=peso_d))


def _compile_mask(wiring_cfg: dict[str, Any]) -> tuple[list[dict], bool]:
    """Compile a wiring config into (mask, random_weights), applying weight overrides."""
    deamon_cfg = wiring_cfg.get("deamon", {})
    if deamon_cfg:
        if "mask" in deamon_cfg:
            mask_id = deamon_cfg["mask"]
            random_weights = get_random_weights(mask_id)
            raw_mask = get_mask(mask_id)
        else:
            random_weights = not deamon_cfg.get("fixed", False)
            raw_mask = compile_deamon_wiring(deamon_cfg)
        exc_w = deamon_cfg.get("excitatory", {}).get("weight")
        inh_w = deamon_cfg.get("inhibitory", {}).get("weight")
    else:
        # Wolfram masks keep mask at wiring top level (no deamon key)
        mask_id = wiring_cfg.get("mask", "")
        random_weights = get_random_weights(mask_id)
        raw_mask = get_mask(mask_id)
        exc_w = None
        inh_w = None
    if exc_w is None and inh_w is None:
        return raw_mask, random_weights

    mask: list[dict] = []
    for d in raw_mask:
        peso = d["peso_dendrita"]
        if exc_w is not None and peso > 0:
            mask.append({**d, "peso_dendrita": exc_w})
        elif inh_w is not None and peso < 0:
            mask.append({**d, "peso_dendrita": inh_w})
        else:
            mask.append(d)
    return mask, random_weights


def _tension_fns_of(cfg: dict[str, Any]) -> list[tuple[str, float]]:
    tf = cfg.get("tension", {}).get("function")
    if tf and isinstance(tf, dict):
        return [(k, float(v)) for k, v in tf.items()]
    return []


def _get_threshold(region_cfg: dict[str, Any]) -> float:
    """Read firing threshold from region config.

    "threshold" is the canonical key; "umbral" is accepted as a legacy alias.
    NeuronaEntrada neurons are never affected — BrainTensor skips them via
    mascara_entrada regardless of what value is stored here.
    """
    return float(region_cfg.get("threshold", region_cfg.get("umbral", 0.0)))


class Experiment(Experimento):
    """Unified NeuroFlow experiment, fully driven by the canonical config."""

    def __init__(self) -> None:
        super().__init__()
        self._config: dict[str, Any] = {}
        self.brain_tensor = None
        self._regions: list[RegionState] = []
        self._regions_by_id: dict[str, RegionState] = {}
        self._connections: list[dict[str, Any]] = []
        self._nerve_circles: list[dict] = []
        self._handler: object | None = None

        # Tissue shortcuts (the first wiring region; used for grid/get_frame/stats)
        self._wiring_region: RegionState | None = None
        self.process_mode: str = "min_vs_max"

        # Learning
        self.learning_enabled: bool = False
        self._lr_per_syn: torch.Tensor | None = None
        self._conn_exclude_range: tuple[float, float] | None = None

        # Spiking
        self.adaptation_enabled: bool = False
        self.up_ticks: int = 5
        self.down_ticks: int = 5

        # Daemon detection
        self._daemon_threshold: float = _DAEMON_THRESHOLD
        self._min_daemon_size: int = _MIN_DAEMON_SIZE
        self._daemon_history: deque[int] = deque(maxlen=_STABILITY_WINDOW)
        self._last_history_gen: int = -1

        self._is_wolfram: bool = False
        self._n_classes: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

    # ── Handler ──

    def _load_handler(self, handler_name: str | None) -> None:
        if not handler_name:
            self._handler = None
            return
        import importlib
        module_name = handler_name.lower()
        try:
            mod = importlib.import_module(f"experiments.handlers.{module_name}")
            cls = getattr(mod, handler_name)
            self._handler = cls()
        except (ImportError, AttributeError) as exc:
            logger.warning("Could not load handler '%s': %s", handler_name, exc)
            self._handler = None

    def _apply_handler_methods(self) -> None:
        """Call handler methods to populate neuron_labels for regions that declare one."""
        if self._handler is None:
            return
        for rs in self._regions:
            if not rs.handler_method:
                continue
            method = getattr(self._handler, rs.handler_method, None)
            if callable(method):
                result = method()
                if isinstance(result, list):
                    rs.neuron_labels = result

    # ── Region helpers ──

    def _wiring_regions(self) -> list[RegionState]:
        return [r for r in self._regions if r.has_wiring]

    def _processed_regions(self) -> list[RegionState]:
        """Regions whose neurons are processed by BrainTensor (all non-NeuronaEntrada)."""
        return [r for r in self._regions if not r.is_entrada]

    def _region_specs(self) -> list[tuple[int, int, str, list, bool]]:
        """Per-region (start, end, mode, fns, soft). Regions without wiring inherit the
        wiring region's process_mode/tension_fns (matching legacy output behaviour)."""
        specs = []
        for r in self._processed_regions():
            if r.has_wiring:
                specs.append((r.start, r.end, r.process_mode, r.tension_fns, r.soft_activation))
            else:
                specs.append((r.start, r.end, self.process_mode, r.tension_fns, r.soft_activation))
        return specs

    def _ascii_region(self) -> RegionState | None:
        return next((r for r in self._regions if r.is_ascii_input), None)

    def _label_target_classes(self, region: RegionState) -> int:
        text = region.source_cfg.get("text", "") if region else ""
        if not text:
            return 0
        if all(t.strip() in _SYNTHETIC_PATTERNS for t in text.split(",")):
            return len([t for t in text.split(",")])
        return len(text)

    @property
    def _wiring_region_id(self) -> str:
        if self._wiring_region is None:
            raise AssertionError("main region not initialized — experiment not built yet")
        return self._wiring_region.id

    @property
    def _input_id(self) -> str | None:
        r = self._ascii_region()
        if r is None:
            r = next((rs for rs in self._regions if rs.source_type == "draw"), None)
        return r.id if r else None

    @property
    def input_enabled(self) -> bool:
        return self._ascii_region() is not None

    @property
    def input_resolution(self) -> int:
        r = self._ascii_region()
        return r.width if r else 0

    @property
    def _input_start_idx(self) -> int:
        r = self._ascii_region()
        return r.start if r else (self._wiring_region.end if self._wiring_region else 0)

    # ── Setup ──

    def setup(self, config: dict[str, Any]) -> None:
        config = _to_canonical(config)
        config = _migrate_config(config)
        self._config = config
        config = _inject_wiring_from_connections(config)  # internal copy with wiring in regions
        self.generation = 0
        self._nerve_circles = []
        self._load_handler(config.get("handler"))

        region_cfgs = config["regions"]
        connections = config.get("connections", [])
        self._connections = connections

        if not any("wiring" in r for r in region_cfgs):
            raise ValueError("config must have at least one region with 'wiring'")

        # ── Spiking / daemon detection from the first wiring region ──
        tissue_cfg = next(r for r in region_cfgs if "wiring" in r)
        spiking_cfg = tissue_cfg.get("spiking")
        self.adaptation_enabled = spiking_cfg is not None
        self.up_ticks = int(spiking_cfg.get("up_ticks", 5)) if spiking_cfg else 5
        self.down_ticks = int(spiking_cfg.get("down_ticks", 5)) if spiking_cfg else 5

        daemon_cfg = tissue_cfg.get("daemon")
        self._daemon_threshold = float(daemon_cfg.get("threshold", _DAEMON_THRESHOLD)) if daemon_cfg else _DAEMON_THRESHOLD
        self._min_daemon_size = int(daemon_cfg.get("min_size", _MIN_DAEMON_SIZE)) if daemon_cfg else _MIN_DAEMON_SIZE

        # Wolfram mode: only when single tissue region with a wolfram mask
        wiring0 = tissue_cfg["wiring"]
        self._is_wolfram = (
            "mask" in wiring0
            and get_mask_type(wiring0["mask"]) == "wolfram"
            and "deamon" not in wiring0
        )

        if self._is_wolfram:
            self._setup_wolfram(tissue_cfg)
            return

        # ── Build RegionStates ──
        # Neuron layout (global tensor) puts the tissue region first so that
        # get_grid/stats can read tissue at [0, width*height). The remaining
        # regions follow in config declaration order. _regions itself keeps
        # declaration order for frame emission.
        self._regions = []
        self._regions_by_id = {}
        for rc in region_cfgs:
            grid = rc.get("grid", {})
            w = int(grid.get("width", 1))
            h = int(grid.get("height", 1))
            source = rc.get("source")
            wiring_cfg = rc.get("wiring") or {}
            rs = RegionState(
                id=rc["id"],
                start=0,
                end=0,
                width=w,
                height=h,
                is_entrada=source is not None and not wiring_cfg,
                source_type=source.get("type") if source else None,
                source_cfg=source or {},
                wiring_cfg=wiring_cfg,
                umbral=_get_threshold(rc),
                process_mode=wiring_cfg.get("process_mode", "min_vs_max"),
                tension_fns=_tension_fns_of(rc),
                soft_activation=rc.get("activation") == "soft",
                neuron_labels=rc.get("neuron_labels"),
                handler_method=rc.get("handler_method"),
            )
            self._regions.append(rs)
            self._regions_by_id[rs.id] = rs

        self._apply_handler_methods()

        self._wiring_region = self._wiring_regions()[0]
        self._mask_type = "kohonen"
        self.process_mode = self._wiring_region.process_mode
        self.width = self._wiring_region.width
        self.height = self._wiring_region.height

        # Assign spans tissue-first, then declaration order
        layout = [self._wiring_region] + [r for r in self._regions if r is not self._wiring_region]
        cursor = 0
        for rs in layout:
            n = rs.width * rs.height
            rs.start = cursor
            rs.end = cursor + n
            cursor += n

        # ── Build neurons (in layout order to match spans) ──
        all_neurons: list[Neurona] = []
        self.regiones = {}
        for rs in layout:
            region_obj = Region(nombre=rs.id)
            for i in range(rs.n):
                if rs.is_entrada:
                    neuron = NeuronaEntrada(id=f"{rs.id}_{i}")
                elif rs is self._wiring_region:
                    x, y = i % rs.width, i // rs.width
                    neuron = Neurona(id=Constructor.key_by_coord(x, y), umbral=rs.umbral)
                else:
                    neuron = Neurona(id=f"{rs.id}_{i}", umbral=rs.umbral)
                all_neurons.append(neuron)
                region_obj.agregar(neuron)
            self.regiones[rs.id] = region_obj

        self.brain = Brain(neuronas=all_neurons)

        # ── Intra-region wiring ──
        for rs in self._wiring_regions():
            mask, random_weights = _compile_mask(rs.wiring_cfg)
            if rs is self._wiring_region:
                centroid_cfg = rs.wiring_cfg.get("deamon", {}).get("centroid") or {}
                centroid_jitter = 1 if centroid_cfg.get("random") else 0
                Constructor().aplicar_mascara_2d(
                    self.brain, rs.width, rs.height, mask,
                    random_weights=random_weights, centroid_jitter=centroid_jitter,
                )
            else:
                neuron_list = list(self.regiones[rs.id].neuronas.values())
                grid2d = [
                    [neuron_list[y * rs.width + x] for x in range(rs.width)]
                    for y in range(rs.height)
                ]
                _apply_mask_to_neuron_grid(grid2d, mask, random_weights)

        # ── Cross-region connections ──
        for conn in connections:
            self._wire_connection(conn)

        # ── Initialization ──
        for rs in self._wiring_regions():
            for neurona in self.regiones[rs.id].neuronas.values():
                neurona.activar_external(random.random())

        # ── Compile ──
        self._compile()

        # ── Source region init ──
        for rs in self._regions:
            if rs.is_ascii_input:
                self._init_ascii_region(rs)

        for rs in self._regions:
            if rs.is_ascii_input:
                self._inject_source(rs)

        self._daemon_history.clear()
        self._last_history_gen = -1

    def _setup_wolfram(self, tissue_cfg: dict[str, Any]) -> None:
        grid = tissue_cfg["grid"]
        self.width = int(grid.get("width", 50))
        self.height = int(grid.get("height", 50))
        self.process_mode = tissue_cfg["wiring"].get("process_mode", "min_vs_max")

        mask_id = tissue_cfg["wiring"]["mask"]
        mask = get_mask(mask_id)
        constructor = Constructor()
        self.brain, self.regiones = constructor.crear_grilla(
            width=self.width, height=self.height,
            filas_entrada=[self.height - 1], filas_salida=[], umbral=0.99,
        )
        constructor.aplicar_mascara_2d(
            self.brain, self.width, self.height, mask,
            random_weights=get_random_weights(mask_id),
        )

        n = self.width * self.height
        self._regions = [RegionState(
            id=tissue_cfg.get("id", "tissue"), start=0, end=n,
            width=self.width, height=self.height, is_entrada=False,
            wiring_cfg=tissue_cfg["wiring"],
            process_mode=self.process_mode,
            tension_fns=_tension_fns_of(tissue_cfg),
        )]
        self._regions_by_id = {self._regions[0].id: self._regions[0]}
        self._connections = []
        self._wiring_region = self._regions[0]

        for neurona in self.brain.neuronas:
            neurona.activar_external(0.0)
        self.brain.get_neurona(
            f"x{self.width // 2}y{self.height - 1}"
        ).activar_external(1.0)

        self._mask_type = "wolfram"
        self._compile()
        self._daemon_history.clear()
        self._last_history_gen = -1

    def _wire_connection(self, conn: dict[str, Any]) -> None:
        if "nerve" in conn:
            self._wire_nerve_connection(conn)
            return
        src = self._regions_by_id.get(conn.get("from"))
        dst = self._regions_by_id.get(conn.get("to"))
        if src is None or dst is None:
            return
        _full = conn.get("full") or {}
        weight = float(_full.get("weight", 0.5))
        density = float(_full.get("density", 1.0))
        src_neurons = list(self.regiones[src.id].neuronas.values())
        dst_neurons = list(self.regiones[dst.id].neuronas.values())
        if not src_neurons or not dst_neurons:
            return

        if (conn.get("type") == "portion" or ("portion" in conn and "full" not in conn)) and conn.get("portion") and src.is_ascii_input:
            self._wire_portion(conn, src, dst, weight, src_neurons, dst_neurons)
            return

        k = max(1, round(len(src_neurons) * density))
        for dst_n in dst_neurons:
            sampled = random.sample(src_neurons, k) if k < len(src_neurons) else src_neurons
            sinapsis_list = [
                Sinapsis(neurona_entrante=s, peso=random.uniform(0.2, 1.0))
                for s in sampled
            ]
            dst_n.dendritas.append(Dendrita(sinapsis=sinapsis_list, peso=weight))

    def _wire_portion(
        self, conn: dict[str, Any], src: RegionState, dst: RegionState,
        weight: float, src_neurons: list, dst_neurons: list,
    ) -> None:
        n_div_y, n_div_x = int(conn["portion"][0]), int(conn["portion"][1])
        res = src.width
        src_by_id = {n.id: n for n in src_neurons}
        src_regions: dict[tuple[int, int], list] = {}
        for ry in range(n_div_y):
            for rx in range(n_div_x):
                py0, py1 = ry * res // n_div_y, (ry + 1) * res // n_div_y
                px0, px1 = rx * res // n_div_x, (rx + 1) * res // n_div_x
                src_regions[(ry, rx)] = [
                    src_by_id[f"{src.id}_{py * res + px}"]
                    for py in range(py0, py1) for px in range(px0, px1)
                ]
        for dst_n in dst_neurons:
            x_str, y_str = dst_n.id[1:].split("y")
            tx, ty = int(x_str), int(y_str)
            rx = min(tx * n_div_x // dst.width, n_div_x - 1)
            ry = min(ty * n_div_y // dst.height, n_div_y - 1)
            sinapsis_list = [
                Sinapsis(neurona_entrante=s, peso=random.uniform(0.2, 1.0))
                for s in src_regions[(ry, rx)]
            ]
            dst_n.dendritas.append(Dendrita(sinapsis=sinapsis_list, peso=weight))

    def _wire_nerve_connection(self, conn: dict[str, Any]) -> None:
        on_rs = self._regions_by_id.get(conn.get("on"))
        if on_rs is None:
            return
        nerve_cfg = conn["nerve"]
        rng = random.Random()

        from_cfg = nerve_cfg.get("from") or {}
        to_cfg = nerve_cfg.get("to") or {}

        circles = place_nerve_circles(nerve_cfg, on_rs.width, on_rs.height, rng)
        to_rs_id = self._regions_by_id.get(to_cfg.get("region"), None)
        from_rs_id = self._regions_by_id.get(from_cfg.get("region"), None) if from_cfg else None
        self._nerve_circles.append({
            "on": on_rs.id,
            "to": to_rs_id.id if to_rs_id else None,
            "from": from_rs_id.id if from_rs_id else None,
            "circles": [{"cx": c.cx, "cy": c.cy, "radius": c.radius} for c in circles],
        })

        if from_cfg:
            from_rs = self._regions_by_id.get(from_cfg.get("region"))
            if from_rs:
                self._wire_nerve_from(circles, on_rs, from_rs, from_cfg)

        if to_cfg:
            to_rs = self._regions_by_id.get(to_cfg.get("region"))
            if to_rs:
                self._wire_nerve_to(circles, on_rs, to_rs, to_cfg)

    def _wire_nerve_from(
        self,
        circles: list[NerveCircle],
        on_rs: RegionState,
        from_rs: RegionState,
        from_cfg: dict[str, Any],
    ) -> None:
        """Wire: from_region neurons → on_region neurons inside circles."""
        density = float(from_cfg.get("density", 0.1))
        weight = float(from_cfg.get("weight", 0.5))
        from_neurons = list(self.regiones[from_rs.id].neuronas.values())
        on_neurons = list(self.regiones[on_rs.id].neuronas.values())
        k = max(1, round(len(from_neurons) * density))
        for circle in circles:
            for x, y, gradient in circle_cells_with_weights(circle, on_rs.width, on_rs.height):
                if random.random() > gradient:
                    continue
                tissue_n = on_neurons[y * on_rs.width + x]
                sampled = random.sample(from_neurons, min(k, len(from_neurons)))
                sinapsis_list = [
                    Sinapsis(neurona_entrante=s, peso=1.0)
                    for s in sampled
                ]
                tissue_n.dendritas.append(Dendrita(sinapsis=sinapsis_list, peso=weight))

    def _wire_nerve_to(
        self,
        circles: list[NerveCircle],
        on_rs: RegionState,
        to_rs: RegionState,
        to_cfg: dict[str, Any],
    ) -> None:
        """Wire: on_region neurons inside circles → to_region neurons."""
        density = float(to_cfg.get("density", 0.1))
        weight = float(to_cfg.get("weight", 0.5))
        to_neurons = list(self.regiones[to_rs.id].neuronas.values())
        on_neurons = list(self.regiones[on_rs.id].neuronas.values())
        for circle in circles:
            participating: list[tuple[Neurona, float]] = []
            for x, y, gradient in circle_cells_with_weights(circle, on_rs.width, on_rs.height):
                if random.random() <= gradient:
                    participating.append((on_neurons[y * on_rs.width + x], gradient))
            if not participating:
                continue
            k = max(1, round(len(participating) * density))
            for to_n in to_neurons:
                sampled = sorted(participating, key=lambda t: t[1], reverse=True)[:k]
                sinapsis_list = [
                    Sinapsis(neurona_entrante=n, peso=grad)
                    for n, grad in sampled
                ]
                to_n.dendritas.append(Dendrita(sinapsis=sinapsis_list, peso=weight))

    def _compile(self) -> None:
        if self._is_wolfram:
            self.brain_tensor = ConstructorTensor.compilar(
                self.brain,
                max_active_steps=self.up_ticks,
                refractory_steps=self.down_ticks,
                adaptation_enabled=self.adaptation_enabled,
                process_mode=self.process_mode,
                tension_fns=self._wiring_region.tension_fns if self._wiring_region else [],
            )
            self.learning_enabled = False
            self._lr_per_syn = None
            return

        region_specs = self._region_specs()
        conn_spans: list[tuple[int, int, int, int]] = []
        for c in self._connections:
            if "nerve" in c:
                on_rs = self._regions_by_id.get(c.get("on"))
                if on_rs is None:
                    continue
                nerve_cfg = c["nerve"]
                from_cfg = nerve_cfg.get("from") or {}
                to_cfg = nerve_cfg.get("to") or {}
                from_rs = self._regions_by_id.get(from_cfg.get("region")) if from_cfg else None
                to_rs = self._regions_by_id.get(to_cfg.get("region")) if to_cfg else None
                if from_rs:
                    conn_spans.append((from_rs.start, from_rs.end, on_rs.start, on_rs.end))
                if to_rs:
                    conn_spans.append((on_rs.start, on_rs.end, to_rs.start, to_rs.end))
            else:
                s = self._regions_by_id.get(c.get("from"))
                d = self._regions_by_id.get(c.get("to"))
                if s is not None and d is not None:
                    conn_spans.append((s.start, s.end, d.start, d.end))

        self.brain_tensor = ConstructorTensor.compilar(
            self.brain,
            max_active_steps=self.up_ticks,
            refractory_steps=self.down_ticks,
            adaptation_enabled=self.adaptation_enabled,
            process_mode=self.process_mode,
            tension_fns=self._wiring_region.tension_fns if self._wiring_region else [],
            connections=conn_spans,
            region_specs=region_specs,
        )

        for rs in self._regions:
            if rs.source_type == "draw":
                rs.draw_base = torch.zeros(rs.n, device=self.brain_tensor.valores.device)

        self._rebuild_lr_per_syn()

    def _rebuild_lr_per_syn(self) -> None:
        """Build [NR, max_syn] effective learning rate per synapse.

        Intra-region synapses use region.wiring.learning_rate; cross-region synapses
        use connection.learning.rate. NeuronaEntrada have no synapses, so they never
        appear. Recomputed from scratch on every soft update.
        """
        bt = self.brain_tensor
        if bt is None:
            return
        NR = bt.n_real
        max_syn = bt.pesos_sinapsis.shape[1]
        lr = torch.zeros(NR, max_syn, dtype=torch.float32, device=bt.device)
        src_safe = bt.indices_fuente.clamp(0, bt.mascara_entrada.shape[0] - 1)
        dst_idx = torch.arange(NR, device=bt.device).unsqueeze(1).expand(NR, max_syn)

        # Intra-region wiring synapses
        for r in self._wiring_regions():
            wlr = r.wiring_cfg.get("learning_rate")
            if not wlr:
                continue
            in_region = (dst_idx >= r.start) & (dst_idx < r.end)
            local = in_region & ~bt.es_cross_region & bt.mascara_valida
            lr = torch.where(local, torch.full_like(lr, float(wlr)), lr)

        # Cross-region connection synapses
        exclude_range: tuple[float, float] | None = None
        for c in self._connections:
            if "nerve" in c:
                nerve_cfg = c["nerve"]
                on_rs = self._regions_by_id.get(c.get("on"))
                if on_rs is None:
                    continue
                for side_key in ("from", "to"):
                    side = nerve_cfg.get(side_key) or {}
                    learning = side.get("learning")
                    if not learning:
                        continue
                    rate = float(learning.get("rate", 0.0))
                    if rate == 0.0:
                        continue
                    other_rs = self._regions_by_id.get(side.get("region"))
                    if other_rs is None:
                        continue
                    if side_key == "from":
                        dst_in = (dst_idx >= on_rs.start) & (dst_idx < on_rs.end)
                        src_in = (src_safe >= other_rs.start) & (src_safe < other_rs.end)
                    else:
                        dst_in = (dst_idx >= other_rs.start) & (dst_idx < other_rs.end)
                        src_in = (src_safe >= on_rs.start) & (src_safe < on_rs.end)
                    conn_mask = dst_in & src_in & bt.mascara_valida
                    lr = torch.where(conn_mask, torch.full_like(lr, rate), lr)
                continue
            s = self._regions_by_id.get(c.get("from"))
            d = self._regions_by_id.get(c.get("to"))
            if s is None or d is None:
                continue
            _full = c.get("full") or {}
            learning = _full.get("learning")
            if isinstance(learning, dict):
                rate = float(learning.get("rate", 0.0))
                er = learning.get("exclude_range")
                if er:
                    exclude_range = (float(er[0]), float(er[1]))
            else:
                rate = float(_full.get("learning_rate", 0.0))
            if rate == 0.0:
                continue
            dst_in = (dst_idx >= d.start) & (dst_idx < d.end)
            src_in = (src_safe >= s.start) & (src_safe < s.end)
            conn_mask = dst_in & src_in & bt.mascara_valida
            lr = torch.where(conn_mask, torch.full_like(lr, rate), lr)

        self._lr_per_syn = lr
        self._conn_exclude_range = exclude_range
        self.learning_enabled = bool((lr != 0).any().item())

    # ── Source injection ──

    def _init_ascii_region(self, rs: RegionState) -> None:
        rs.char_index = 0
        rs.frame_in_char = 0
        rs.in_gap = False
        rs.char_images = {}
        text = rs.source_cfg.get("text", "") or ""
        if text and not self._is_synthetic(text):
            font_id = rs.source_cfg.get("font", "press_start_2p")
            font_size = int(rs.source_cfg.get("font_size", 10))
            for char in set(text):
                rs.char_images[char] = render_char(char, rs.width, font_id=font_id, font_size=font_size)

    @staticmethod
    def _is_synthetic(text: str) -> bool:
        if not text:
            return False
        return all(tok.strip() in _SYNTHETIC_PATTERNS for tok in text.split(","))

    @staticmethod
    def _make_synthetic(name: str, res: int) -> np.ndarray:
        frame = np.zeros((res, res), dtype=np.float64)
        if name == "HALF_TOP":
            frame[: res // 2, :] = 1.0
        elif name == "HALF_BOT":
            frame[res // 2 :, :] = 1.0
        elif name == "BARS_H":
            for i in range(3):
                y = int(res * (i + 1) / 4)
                frame[max(0, y - 1):min(res, y + 2), :] = 1.0
        elif name == "BARS_V":
            for i in range(3):
                x = int(res * (i + 1) / 4)
                frame[:, max(0, x - 1):min(res, x + 2)] = 1.0
        elif name == "DOT_TL":
            frame[:5, :5] = 1.0
        elif name == "DOT_BR":
            frame[-5:, -5:] = 1.0
        return frame

    def _inject_source(self, rs: RegionState) -> None:
        if rs.source_type == "ascii":
            self._inject_ascii(rs)
        elif rs.source_type == "error_diff":
            self._inject_error_diff(rs)

    def _inject_ascii(self, rs: RegionState) -> None:
        res = rs.width
        text = rs.source_cfg.get("text", "") or ""
        noise = rs.source_cfg.get("noise") or {}
        background = float(noise.get("background", 0.0))
        shift = bool(noise.get("shift", False))

        if not text or rs.in_gap:
            frame = self._rng.integers(0, 2, size=(res, res)).astype(np.float64)
        elif self._is_synthetic(text):
            tokens = [t.strip() for t in text.split(",")]
            frame = self._make_synthetic(tokens[rs.char_index % len(tokens)], res)
            if background > 0:
                frame = apply_white_noise(frame, noise_prob=background, rng=self._rng)
        else:
            frame = rs.char_images[text[rs.char_index]].copy()
            if shift:
                frame = apply_shift_noise(frame, self._rng)
            if background > 0:
                frame = apply_white_noise(frame, noise_prob=background, rng=self._rng)

        rs.current_frame = frame
        if self.brain_tensor is not None:
            flat = torch.from_numpy(frame.flatten()).float()
            self.brain_tensor.valores[rs.start : rs.start + len(flat)] = flat

    def _inject_error_diff(self, rs: RegionState) -> None:
        if self.brain_tensor is None:
            return
        n = rs.n
        out_id = rs.source_cfg.get("label_region") or rs.source_cfg.get("output_region")
        out_region = self._regions_by_id.get(out_id) if out_id else None
        if out_region is None:
            out_region = next((r for r in self._wiring_regions() if r is not self._wiring_region), None)
        if out_region is None:
            return

        pred_full = self.brain_tensor.valores[out_region.start:out_region.end].cpu()
        pred = self._compress_to_n(pred_full, n)

        target_id = rs.source_cfg.get("target", "input")
        target: torch.Tensor | None = None
        if target_id == "label":
            ascii_r = self._ascii_region()
            classes = max(1, self._label_target_classes(ascii_r)) if ascii_r else 1
            class_idx = (ascii_r.char_index % n) if ascii_r else 0
            lbl = torch.zeros(n)
            lbl[class_idx] = 1.0
            target = lbl
        else:
            target_region = self._regions_by_id.get(target_id) or self._ascii_region()
            if target_region is not None and target_region.current_frame is not None:
                flat = torch.from_numpy(target_region.current_frame.flatten()).float()
                target = self._compress_to_n(flat, n)

        if target is None:
            return

        if rs.source_cfg.get("diff_mode", "abs") == "relu":
            error = torch.relu(target - pred)
        else:
            error = torch.abs(target - pred)
        self.brain_tensor.valores[rs.start:rs.start + n] = error.to(self.brain_tensor.device)

    def _inject_label_mismatch(self, rs: RegionState) -> None:
        """Nociceptor injection — two modes controlled by source.mode:

        "mismatch" (default): neuron i fires with label_region[i]'s activation
          when its label ≠ current char, else 0.
        "active_avg": signal = mean of all label_region activations (0-1),
          broadcast uniformly to all nociceptor neurons.
        """
        if self.brain_tensor is None:
            return
        label_r = self._regions_by_id.get(rs.source_cfg.get("label_region", ""))
        if label_r is None:
            return

        mode = rs.source_cfg.get("mode", "mismatch")

        if mode == "active_avg":
            label_vals = self.brain_tensor.valores[label_r.start : label_r.end].cpu()
            avg = label_vals.clamp(0.0, 1.0).mean().item()
            values = torch.full((rs.n,), avg)
            self.brain_tensor.valores[rs.start : rs.start + rs.n] = values.to(self.brain_tensor.device)
            return

        # mode == "mismatch" (default)
        char_r = self._regions_by_id.get(rs.source_cfg.get("char_region", "input"))
        if char_r is None or not label_r.neuron_labels:
            return
        text = char_r.source_cfg.get("text", "")
        if not text:
            return
        current_char = text[char_r.char_index % len(text)]
        label_vals = self.brain_tensor.valores[label_r.start : label_r.end].cpu()
        values = torch.zeros(rs.n)
        flat_idx = 0
        for row_labels in label_r.neuron_labels:
            for label in row_labels:
                if flat_idx < rs.n:
                    target = 1.0 if label == current_char else 0.0
                    values[flat_idx] = abs(target - label_vals[flat_idx].clamp(0, 1).item())
                flat_idx += 1
        self.brain_tensor.valores[rs.start : rs.start + rs.n] = values.to(self.brain_tensor.device)

    @staticmethod
    def _compress_to_n(t: torch.Tensor, n: int) -> torch.Tensor:
        length = len(t)
        if length == n:
            return t
        if length > n:
            chunk = length // n
            result = torch.zeros(n)
            for i in range(n):
                s = i * chunk
                e = (i + 1) * chunk if i < n - 1 else length
                result[i] = t[s:e].mean()
            return result
        result = torch.zeros(n)
        result[:length] = t
        return result

    def _current_label(self, region: RegionState) -> torch.Tensor | None:
        """One-hot label for a label-source region (n = region size)."""
        ascii_r = self._ascii_region()
        if ascii_r is None or region.n == 0:
            return None
        label = torch.zeros(region.n)
        label[ascii_r.char_index % region.n] = 1.0
        return label

    def _label_region(self) -> RegionState | None:
        return next((r for r in self._regions if r.source_type == "label"), None)

    # ── Processing ──

    def step(self) -> dict[str, Any]:
        for rs in self._regions:
            if rs.is_ascii_input:
                self._inject_ascii(rs)

        self._apply_draw_noise()
        self.brain_tensor.procesar()

        label_region = self._label_region()
        if label_region is not None:
            label = self._current_label(label_region)
            if label is not None:
                label_tension = label * 2.0 - 1.0
                self.brain_tensor.tensiones[label_region.start:label_region.end] = (
                    label_tension.to(self.brain_tensor.device)
                )

        for rs in self._regions:
            if rs.source_type == "error_diff":
                self._inject_error_diff(rs)
            elif rs.source_type == "label_mismatch":
                self._inject_label_mismatch(rs)

        if self.learning_enabled and self._lr_per_syn is not None:
            self.brain_tensor.learn(
                lr_per_syn=self._lr_per_syn,
                conn_exclude_range=self._conn_exclude_range,
            )

        self.generation += 1
        self._advance_ascii_frames()

        return {"type": "frame", "generation": self.generation}

    @staticmethod
    def _draw_noise_prob(source_cfg: dict[str, Any]) -> float:
        """Read noise probability from source config.

        Accepts both flat ``"noise": 0.5`` and nested ``"noise": {"background": 0.5}``
        (same format as ASCII noise.background).
        """
        noise_val = source_cfg.get("noise", 0.0)
        if isinstance(noise_val, dict):
            return float(noise_val.get("background", 0.0))
        return float(noise_val)

    def _apply_draw_noise(self) -> None:
        """Inject background noise into draw regions using draw_base as clean source.

        Uses draw_base (the user's painted values) to apply white noise each step.
        The noisy frame is written to brain_tensor so it is visible in the display
        and processed by the network — the same pattern as ASCII noise.background.
        draw_base itself is never mutated here; it is only updated by paint().
        """
        for rs in self._regions:
            if rs.source_type != "draw" or rs.draw_base is None:
                continue
            noise_prob = self._draw_noise_prob(rs.source_cfg)
            if noise_prob <= 0.0:
                continue
            flat = rs.draw_base.cpu().numpy().astype(np.float64)
            noisy = apply_white_noise(flat, noise_prob=noise_prob, rng=self._rng)
            self.brain_tensor.valores[rs.start:rs.end] = torch.from_numpy(
                noisy.astype(np.float32)
            ).to(self.brain_tensor.valores.device)

    def _advance_ascii_frames(self) -> None:
        for rs in self._regions:
            if not rs.is_ascii_input:
                continue
            text = rs.source_cfg.get("text", "") or ""
            if not text:
                continue
            noise = rs.source_cfg.get("noise") or {}
            inter_char = bool(noise.get("inter_char", False))
            fpc = max(1, int(rs.source_cfg.get("frames_per_char", 10)))
            n_items = (
                len([t.strip() for t in text.split(",")])
                if self._is_synthetic(text) else len(text)
            )
            rs.frame_in_char += 1
            if rs.in_gap:
                if rs.frame_in_char >= fpc:
                    rs.in_gap = False
                    rs.frame_in_char = 0
                    rs.char_index = (rs.char_index + 1) % n_items
            elif rs.frame_in_char >= fpc:
                if inter_char:
                    rs.in_gap = True
                    rs.frame_in_char = 0
                else:
                    rs.frame_in_char = 0
                    rs.char_index = (rs.char_index + 1) % n_items

    def paint(
        self,
        region_id: str | None,
        cells: list[dict[str, int]],
        value: float,
    ) -> None:
        """Paint cells on a draw region, keeping draw_base in sync with brain_tensor."""
        if self.brain_tensor is None:
            return
        region = self._regions_by_id.get(region_id) if region_id else self._wiring_region
        if region is None:
            region = self._wiring_region
        if region is None:
            return
        rw, rh, start = region.width, region.height, region.start
        for cell in cells:
            x, y = cell.get("x", 0), cell.get("y", 0)
            if 0 <= x < rw and 0 <= y < rh:
                idx = start + y * rw + x
                if idx < self.brain_tensor.n_real:
                    self.brain_tensor.set_valor(idx, value)
                    if region.source_type == "draw" and region.draw_base is not None:
                        region.draw_base[idx - start] = value

    def step_n(self, count: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for _ in range(count):
            result = self.step()
        return result

    def click(self, x: int, y: int) -> None:
        if self.brain_tensor is None or self._wiring_region is None:
            return
        idx = y * self.width + x
        ascii_r = self._ascii_region()
        limit = ascii_r.start if ascii_r else self.brain_tensor.n_real
        if 0 <= idx < limit:
            current = self.brain_tensor.valores[idx].item()
            self.brain_tensor.set_valor(idx, 0.0 if current >= 0.5 else 1.0)

    # ── Frame serialization ──

    def get_frame(self) -> list[list[float]]:
        if self.brain_tensor:
            return self.brain_tensor.get_grid(self.width, self.height)
        return super().get_frame()

    def get_tension_frame(self) -> list[list[float]] | None:
        if self.brain_tensor:
            return self.brain_tensor.get_tension_grid(self.width, self.height)
        return None

    def _region_values(self, rs: RegionState) -> torch.Tensor:
        return self.brain_tensor.valores[rs.start:rs.end]

    def get_region_frames(self) -> dict[str, list[list[float]]]:
        if self.brain_tensor is None:
            return {}
        result: dict[str, list[list[float]]] = {}
        for rs in self._regions:
            vals = self._region_values(rs).reshape(rs.height, rs.width).tolist()
            if rs.is_entrada and rs.source_type not in (None, "draw"):
                result[rs.id] = [[round(v, 3) for v in row] for row in vals]
            elif rs.soft_activation:
                result[rs.id] = [[round(v, 3) for v in row] for row in vals]
            else:
                result[rs.id] = [[round(v) for v in row] for row in vals]
        return result

    def get_region_tension_frames(self) -> dict[str, list[list[float]]]:
        if self.brain_tensor is None:
            return {}
        result: dict[str, list[list[float]]] = {}
        for rs in self._processed_regions():
            t = self.brain_tensor.tensiones[rs.start:rs.end]
            result[rs.id] = [
                [round(v, 3) for v in row]
                for row in t.reshape(rs.height, rs.width).tolist()
            ]
        # Emit in config declaration order
        return {r.id: result[r.id] for r in self._regions if r.id in result}

    def get_label_frames(self) -> dict[str, list[list[float]]] | None:
        label_region = self._label_region()
        if label_region is None:
            return None
        label = self._current_label(label_region)
        if label is None:
            return None
        grid = label.reshape(label_region.height, label_region.width).tolist()
        return {label_region.id: [[round(v) for v in row] for row in grid]}

    def get_neuron_label_map(self) -> dict[str, list[list[str]]] | None:
        result = {r.id: r.neuron_labels for r in self._regions if r.neuron_labels}
        return result if result else None

    def get_input_frame(self) -> list[list[float]] | None:
        rs = self._ascii_region()
        if rs is not None and rs.current_frame is not None:
            return rs.current_frame.tolist()
        return None

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        if self.brain_tensor is None:
            return super().get_stats()

        n_tissue = self.width * self.height
        vals = self.brain_tensor.valores[:n_tissue]
        active = int((vals > self._daemon_threshold).sum().item())

        count, daemon_indices, noise_indices, sizes = _detect_daemons(
            self.brain_tensor.valores, self.width, self.height,
            self._daemon_threshold, self._min_daemon_size,
        )
        avg_size = round(sum(sizes) / len(sizes), 1) if sizes else 0.0

        if daemon_indices:
            daemon_mask = torch.zeros(n_tissue, dtype=torch.bool)
            daemon_mask[list(daemon_indices)] = True
            inside_mean = vals[daemon_mask].mean().item()
            outside = vals[~daemon_mask]
            outside_mean = outside.mean().item() if outside.numel() > 0 else 0.0
            exclusion = inside_mean - outside_mean
        else:
            exclusion = 0.0

        if self.generation != self._last_history_gen:
            self._daemon_history.append(count)
            self._last_history_gen = self.generation

        if len(self._daemon_history) >= 2:
            counts = list(self._daemon_history)
            mean_c = sum(counts) / len(counts)
            if mean_c > 0:
                variance = sum((c - mean_c) ** 2 for c in counts) / len(counts)
                cv = (variance ** 0.5) / mean_c
                stability = round(max(0.0, min(1.0, 1.0 - cv)), 3)
            else:
                stability = 1.0 if all(c == 0 for c in counts) else 0.0
        else:
            stability = 0.0

        stats: dict[str, Any] = {
            "active_cells": active,
            "steps": self.generation,
            "daemon_count": count,
            "avg_daemon_size": avg_size,
            "noise_cells": len(noise_indices),
            "stability": stability,
            "exclusion": round(exclusion, 3),
        }

        ascii_r = self._ascii_region()
        if ascii_r is not None:
            text = ascii_r.source_cfg.get("text", "") or ""
            if not text:
                current_char = ""
            elif ascii_r.in_gap:
                current_char = "gap"
            elif self._is_synthetic(text):
                tokens = [t.strip() for t in text.split(",")]
                current_char = tokens[ascii_r.char_index % len(tokens)]
            else:
                current_char = text[ascii_r.char_index]
            stats.update({
                "current_char": current_char,
                "char_index": ascii_r.char_index,
                "frame_in_char": ascii_r.frame_in_char,
                "frames_per_char": max(1, int(ascii_r.source_cfg.get("frames_per_char", 10))),
                "input_resolution": ascii_r.width,
            })

        return stats

    # ── Inspect ──

    def _region_at(self, region_id: str | None) -> RegionState:
        if region_id and region_id in self._regions_by_id:
            return self._regions_by_id[region_id]
        return self._wiring_region

    def inspect(self, x: int, y: int, region_id: str | None = None) -> dict[str, Any]:
        if self.brain_tensor is None:
            result = super().inspect(x, y)
            result["input_weight_grid"] = None
            result["region_id"] = region_id
            return result

        target = self._region_at(region_id)
        is_wiring_region = target is self._wiring_region
        neuron_idx = target.start + y * target.width + x

        sources = self.brain_tensor.indices_fuente[neuron_idx]
        weights = self.brain_tensor.pesos_sinapsis[neuron_idx]
        dend_weights = self.brain_tensor.pesos_dendrita[neuron_idx]
        valid = self.brain_tensor.mascara_valida[neuron_idx]
        dend_ids = self.brain_tensor.dendrita_ids[neuron_idx]

        total_sinapsis = int(valid.sum().item())
        total_dendritas = int(dend_ids[valid].unique().numel()) if total_sinapsis > 0 else 0

        # Per-source-region effective weight maps
        per_region: dict[str, dict[int, float]] = {r.id: {} for r in self._regions}
        for i in range(sources.shape[0]):
            if not valid[i]:
                continue
            src = sources[i].item()
            w = weights[i].item()
            dw = dend_weights[i].item()
            src_region = self._region_of_index(src)
            if src_region is None:
                continue
            local_src = src - src_region.start
            if src_region.is_ascii_input:
                per_region[src_region.id][local_src] = w
            else:
                acc = per_region[src_region.id]
                acc[local_src] = max(-1.0, min(1.0, acc.get(local_src, 0.0) + w * dw))

        wiring_pesos = per_region.get(self._wiring_region.id, {})

        # When inspecting a nerve `from` region (e.g. nociceptor), its outgoing synapses
        # live in tissue neurons' dendrites, not in its own. Compute the reverse footprint:
        # which tissue neurons receive from this specific neuron, and with what weight.
        is_nerve_from = (
            not is_wiring_region
            and self._wiring_region is not None
            and any(entry.get("from") == target.id for entry in self._nerve_circles)
        )
        if is_nerve_from:
            bt = self.brain_tensor
            on_rs = self._wiring_region
            t_size = on_rs.n
            t_start = on_rs.start
            srcs_t = bt.indices_fuente[t_start : t_start + t_size]
            ws_t = bt.pesos_sinapsis[t_start : t_start + t_size]
            dws_t = bt.pesos_dendrita[t_start : t_start + t_size]
            valid_t = bt.mascara_valida[t_start : t_start + t_size]
            hit = (srcs_t == neuron_idx) & valid_t
            eff = (ws_t * dws_t * hit.float()).sum(dim=1).cpu().tolist()
            max_abs = max(abs(v) for v in eff) if eff else 0.0
            if max_abs > 1e-9:
                inv = 1.0 / max_abs
                wiring_pesos = {
                    r * on_rs.width + c: round(eff[r * on_rs.width + c] * inv, 4)
                    for r in range(on_rs.height)
                    for c in range(on_rs.width)
                    if abs(eff[r * on_rs.width + c]) > 1e-12
                }

        weight_grid: list[list[float | None]] = []
        for row in range(self.height):
            fila: list[float | None] = []
            for col in range(self.width):
                if is_wiring_region and col == x and row == y:
                    fila.append(999)
                else:
                    fila.append(wiring_pesos.get(row * self.width + col))
            weight_grid.append(fila)

        activation = self.brain_tensor.valores[neuron_idx].item()
        tension = self.brain_tensor.tensiones[neuron_idx].item()

        target_linear_idx = y * target.width + x
        region_nerve_circles: list[dict] = []
        for entry in self._nerve_circles:
            if entry["on"] != self._wiring_region_id:
                continue
            # When inspecting a non-tissue region, only include circles wired to/from it
            if not is_wiring_region:
                to_id = entry.get("to")
                from_id = entry.get("from")
                if to_id != target.id and from_id != target.id:
                    continue
            region_nerve_circles.extend(entry["circles"])

        result: dict[str, Any] = {
            "type": "connections",
            "x": x,
            "y": y,
            "region_id": target.id,
            "activation": round(activation, 4),
            "tension": round(tension, 4),
            "total_dendritas": total_dendritas,
            "total_sinapsis": total_sinapsis,
            "weight_grid": weight_grid,
        }
        if region_nerve_circles:
            result["nerve_circles"] = region_nerve_circles

        ascii_r = self._ascii_region()
        if ascii_r is not None and is_wiring_region:
            res = ascii_r.width
            inp = per_region.get(ascii_r.id, {})
            grid: list[list[float]] = []
            for r in range(ascii_r.height):
                grid.append([inp.get(r * res + c, 0.0) for c in range(res)])
            result["input_weight_grid"] = grid
            result["input_weight_width"] = res
            result["input_weight_height"] = ascii_r.height
        else:
            result["input_weight_grid"] = None

        # Source-region footprints on tissue canvas, keyed by nociceptor region ID.
        # Each entry: {"grid": ..., "density": max_abs} — grid normalized to [-1, 1].
        # Frontend sorts by density descending and renders highest-density first so all layers stay visible.
        noc_regions = [r for r in self._regions if r.source_type in _NOCICEPTOR_SOURCE_TYPES]
        if noc_regions and self._wiring_region is not None and is_wiring_region:
            on_rs = self._wiring_region
            region_overlays: dict[str, dict] = {}
            bt = self.brain_tensor
            t_size = on_rs.n
            t_start = on_rs.start
            srcs_t  = bt.indices_fuente[t_start : t_start + t_size]
            ws_t    = bt.pesos_sinapsis[t_start : t_start + t_size]
            dws_t   = bt.pesos_dendrita[t_start : t_start + t_size]
            valid_t = bt.mascara_valida[t_start : t_start + t_size]
            for noc_rs in noc_regions:
                noc_mask = (srcs_t >= noc_rs.start) & (srcs_t < noc_rs.end) & valid_t
                eff = ws_t * dws_t * noc_mask.float()
                noc_flat = eff.sum(dim=1).cpu().tolist()
                max_abs = max(abs(v) for v in noc_flat) if noc_flat else 0.0
                if max_abs <= 1e-9:
                    continue
                inv = 1.0 / max_abs
                noc_grid: list[list[float | None]] = []
                for r in range(on_rs.height):
                    row_g: list[float | None] = []
                    for c in range(on_rs.width):
                        v = noc_flat[r * on_rs.width + c]
                        row_g.append(round(v * inv, 4) if abs(v) > 1e-12 else None)
                    noc_grid.append(row_g)
                region_overlays[noc_rs.id] = {"grid": noc_grid, "density": round(max_abs, 6)}
            if region_overlays:
                result["region_overlays"] = region_overlays

        # Nerve 'from' regions — when inspecting wiring region their contribution is already
        # captured by region_overlays and nerve circles; a 1×1 source grid is redundant.
        nerve_from_ids = {entry.get("from") for entry in self._nerve_circles if entry.get("from")}

        # Generic per-source-region grids (output / etc.)
        for src_region in self._regions:
            if src_region is self._wiring_region or src_region is ascii_r:
                continue
            if src_region is target:
                continue
            if is_wiring_region and src_region.id in nerve_from_ids:
                continue
            pesos = per_region.get(src_region.id)
            if not pesos:
                continue
            sw, sh = src_region.width, src_region.height
            grid2: list[list[float | None]] = []
            for r in range(sh):
                grid2.append([pesos.get(r * sw + c) for c in range(sw)])
            key = src_region.id
            result[f"{key}_weight_grid"] = grid2
            result[f"{key}_weight_width"] = sw
            result[f"{key}_weight_height"] = sh

        return result

    def _region_of_index(self, idx: int) -> RegionState | None:
        for r in self._regions:
            if r.start <= idx < r.end:
                return r
        return None

    # ── Update config ──

    def update_config(self, config: dict[str, Any]) -> bool:
        if self.brain_tensor is None:
            return False

        new_config = _to_canonical(config)
        new_config = _migrate_config(new_config)
        new_internal = _inject_wiring_from_connections(new_config)
        old_internal = _inject_wiring_from_connections(self._config)

        old_regions = {r.get("id"): r for r in old_internal.get("regions", [])}
        new_regions = {r.get("id"): r for r in new_internal.get("regions", [])}

        needs_reconnect = set(old_regions) != set(new_regions)

        if not needs_reconnect:
            for rid, new_r in new_regions.items():
                old_r = old_regions[rid]
                if new_r.get("grid") != old_r.get("grid"):
                    needs_reconnect = True
                    break
                old_w = old_r.get("wiring") or {}
                new_w = new_r.get("wiring") or {}
                for k in ("mask", "deamon"):
                    if old_w.get(k) != new_w.get(k):
                        needs_reconnect = True
                        break
                if needs_reconnect:
                    break
                old_src = old_r.get("source") or {}
                new_src = new_r.get("source") or {}
                if old_src.get("type") != new_src.get("type"):
                    needs_reconnect = True
                    break

        if not needs_reconnect:
            # Inter-region connection topology (weights / density / from / to) → rebuild
            def conn_key(c: dict) -> tuple:
                if "nerve" in c:
                    nerve = c["nerve"]
                    return (
                        "nerve", c.get("on"),
                        nerve.get("insertion"), nerve.get("count"),
                        nerve.get("radius"),
                        nerve.get("paired", False),
                        (nerve.get("from") or {}).get("region"),
                        (nerve.get("to") or {}).get("region"),
                    )
                _full = c.get("full") or {}
                return (c.get("from"), c.get("to"),
                        "full" if "full" in c else c.get("type"),
                        _full.get("weight"),
                        _full.get("density"),
                        tuple(c.get("portion") or ()))
            old_conns = [conn_key(c) for c in old_internal.get("connections", [])]
            new_conns = [conn_key(c) for c in new_internal.get("connections", [])]
            if old_conns != new_conns:
                needs_reconnect = True

        if needs_reconnect:
            self.setup(new_config)
            return False

        # ── Soft updates ──
        for rs in self._regions:
            new_r = new_regions.get(rs.id, {})
            new_wiring = new_r.get("wiring") or {}
            new_source = new_r.get("source") or {}

            if rs.has_wiring:
                rs.wiring_cfg = new_wiring
                rs.process_mode = new_wiring.get("process_mode", "min_vs_max")
            if not rs.is_entrada:
                rs.tension_fns = _tension_fns_of(new_r)
                rs.soft_activation = new_r.get("activation") == "soft"

            if not rs.is_entrada:
                new_umbral = _get_threshold(new_r)
                if new_umbral != rs.umbral:
                    rs.umbral = new_umbral
                    self.brain_tensor.umbrales[rs.start:rs.end] = new_umbral

            if rs.is_ascii_input and new_source:
                self._soft_update_ascii(rs, new_source)
                rs.source_cfg = new_source
            elif rs.source_type == "draw" and new_source:
                rs.source_cfg = new_source

        # Sync brain_tensor global mode/specs/spiking from tissue
        if self._wiring_region is not None:
            self.process_mode = self._wiring_region.process_mode
            self.brain_tensor.process_mode = self.process_mode
            self.brain_tensor.tension_fns = self._wiring_region.tension_fns
        self.brain_tensor.region_specs = self._region_specs()
        self.brain_tensor.soft_mask = self.brain_tensor._build_soft_mask()

        new_wiring_cfg = new_regions.get(self._wiring_region.id, {}) if self._wiring_region else {}
        new_spiking = new_wiring_cfg.get("spiking")
        if new_spiking:
            self.adaptation_enabled = True
            self.up_ticks = int(new_spiking.get("up_ticks", self.up_ticks))
            self.down_ticks = int(new_spiking.get("down_ticks", self.down_ticks))
            self.brain_tensor.adaptation_enabled = True
            self.brain_tensor.max_active_steps = self.up_ticks
            self.brain_tensor.refractory_steps = self.down_ticks
        else:
            self.adaptation_enabled = False
            self.brain_tensor.adaptation_enabled = False

        # learning rates may have changed (intra-region wiring lr / connection lr)
        self._connections = new_internal.get("connections", [])
        self._rebuild_lr_per_syn()

        self._config = new_config  # store external format for frontend
        return True

    def _soft_update_ascii(self, rs: RegionState, new_source: dict[str, Any]) -> None:
        old_source = rs.source_cfg
        font_changed = False
        new_font = new_source.get("font", old_source.get("font", "press_start_2p"))
        new_font_size = int(new_source.get("font_size", old_source.get("font_size", 10)))
        if new_font != old_source.get("font", "press_start_2p"):
            font_changed = True
        if new_font_size != int(old_source.get("font_size", 10)):
            font_changed = True

        text_changed = False
        new_text = new_source.get("text", old_source.get("text", "")) or ""
        if new_text != (old_source.get("text", "") or ""):
            text_changed = True
            rs.char_index = 0
            rs.frame_in_char = 0
            rs.in_gap = False

        if (font_changed or text_changed) and not self._is_synthetic(new_text):
            rs.char_images = {}
            for char in set(new_text):
                rs.char_images[char] = render_char(char, rs.width, font_id=new_font, font_size=new_font_size)

    def reset(self) -> None:
        self.setup(self._config)

    def is_complete(self) -> bool:
        return False

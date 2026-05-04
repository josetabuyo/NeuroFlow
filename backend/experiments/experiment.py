"""Unified NeuroFlow experiment.

Supports all features through opt-in config sections:
  - regions[].wiring (required: at least one region with wiring)
  - regions[].source (optional: ASCII/synthetic input stream)
  - regions[].noise (optional: background, shift, inter-char)
  - regions[].spiking (optional: spike frequency adaptation)
  - connections[].learning_rate (optional: Hebbian updates on inter-region dendrites)
  - regions[].wiring.learning_rate (optional: Hebbian updates on intra-region dendrites)

Config is canonical JSON (regions + connections). Legacy flat format (grid/wiring/input/
learning/noise/spiking at top level) is auto-converted via _to_canonical().
"""

from __future__ import annotations

import logging
import random
from collections import deque
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
from core.ascii_renderer import render_char, apply_white_noise, apply_shift_noise
from .base import Experimento

logger = logging.getLogger(__name__)

_STABILITY_WINDOW = 20
_DAEMON_THRESHOLD = 0.5
_MIN_DAEMON_SIZE = 3


def _detect_daemons(
    values: torch.Tensor,
    width: int,
    height: int,
    threshold: float,
    min_size: int = _MIN_DAEMON_SIZE,
) -> tuple[int, set[int], set[int], list[int]]:
    """Detect daemons as connected components of active neurons (8-connectivity).

    Returns (count, daemon_indices, noise_indices, sizes).
    """
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


def _to_canonical(config: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy or already-canonical config to the canonical regions/connections format.

    Already canonical (has 'regions' key) → returned as-is.
    Legacy (has 'grid' key at top level) → converted according to the mapping table.

    Legacy → canonical mapping:
      grid                         → regions[tissue].grid
      wiring.*                     → regions[tissue].wiring.*
      input.resolution             → regions[input].grid = {width, height}
      input.text/frames_per_char/  → regions[input].source.*
        font/font_size/density
      input.dendrite_input_weight  → connections[0].weight
      input.portion                → connections[0].type="portion", .portion=[…]
      learning.rate * lr_input     → connections[0].learning_rate  (if > 0)
      learning.rate * max(lr_exc,  → regions[tissue].wiring.learning_rate (if > 0)
        lr_inh)
      noise.*                      → regions[tissue].noise.*
      spiking.*                    → regions[tissue].spiking.*
      daemon.*                     → regions[tissue].daemon.*
    """
    if "regions" in config:
        return config

    config = {**config}

    # Fill safe defaults for missing required legacy fields
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

    # Apply learning rates: effective_wiring = rate * max(lr_exc, lr_inh)
    if learning_cfg is not None:
        base_rate = float(learning_cfg.get("rate", 1.0))
        lr_exc = float(learning_cfg.get("lr_exc", 1.0))
        lr_inh = float(learning_cfg.get("lr_inh", 1.0))
        effective_wiring = base_rate * max(lr_exc, lr_inh)
        if effective_wiring > 0:
            wiring["learning_rate"] = effective_wiring

    # Build tissue region
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

        regions.append({
            "id": "input",
            "grid": {"width": resolution, "height": resolution},
            "source": source,
        })

        conn: dict[str, Any] = {
            "from": "input",
            "to": "tissue",
            "type": "full",
            "weight": float(input_cfg.get("dendrite_input_weight", 0.2)),
        }
        if "density" in input_cfg:
            conn["density"] = float(input_cfg["density"])
        portion = input_cfg.get("portion")
        if portion is not None:
            conn["type"] = "portion"
            conn["portion"] = list(portion)

        if learning_cfg is not None:
            base_rate = float(learning_cfg.get("rate", 1.0))
            lr_input = float(learning_cfg.get("lr_input", 1.0))
            effective_input = base_rate * lr_input
            if effective_input > 0:
                conn["learning"] = {"rate": effective_input}

        connections.append(conn)

    canonical: dict[str, Any] = {}
    for key in ("name", "description"):
        if key in config:
            canonical[key] = config[key]
    canonical["regions"] = regions
    if connections:
        canonical["connections"] = connections

    return canonical


class Experiment(Experimento):
    """Unified NeuroFlow experiment with opt-in features."""

    _SYNTHETIC_PATTERNS = {
        "HALF_TOP", "HALF_BOT", "BARS_H", "BARS_V", "DOT_TL", "DOT_BR",
    }

    def __init__(self) -> None:
        super().__init__()
        self._config: dict[str, Any] = {}
        self.brain_tensor = None
        self.process_mode: str = "min_vs_max"

        # Input state
        self.input_enabled: bool = False
        self.input_text: str = ""
        self.input_resolution: int = 20
        self.frames_per_char: int = 10
        self.dendrite_input_weight: float = 0.2
        self.input_portion: tuple[int, int] | None = None
        self._font_id: str = "press_start_2p"
        self._font_size: int = 10
        self._char_images: dict[str, np.ndarray] = {}
        self._char_index: int = 0
        self._frame_in_char: int = 0
        self._in_gap: bool = False
        self._current_input_frame: np.ndarray | None = None
        self._input_start_idx: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

        # Noise state
        self.background_noise: float = 0.0
        self.shift_noise: bool = False
        self.inter_char_noise: bool = False

        # Wiring overrides
        self.dendrite_exc_weight: float | None = None
        self.dendrite_inh_weight: float | None = None
        self._tension_fns: list[tuple[str, float]] = []

        # Learning state
        # In canonical mode: lr=1.0 always; lr_exc/lr_inh = wiring.learning_rate;
        # lr_input = connections[0].learning_rate. Set to 0 to freeze.
        self.learning_enabled: bool = False
        self.learning_rate: float = 1.0
        self.lr_exc: float = 0.0
        self.lr_inh: float = 0.0
        self.lr_input: float = 0.0
        self._conn_exclude_range: tuple[float, float] | None = None

        # Spiking state
        self.adaptation_enabled: bool = False
        self.up_ticks: int = 5
        self.down_ticks: int = 5

        # Daemon detection config
        self._daemon_threshold: float = _DAEMON_THRESHOLD
        self._min_daemon_size: int = _MIN_DAEMON_SIZE

        # Daemon stats
        self._daemon_history: deque[int] = deque(maxlen=_STABILITY_WINDOW)
        self._last_history_gen: int = -1

    def setup(self, config: dict[str, Any]) -> None:
        """Build the network from a config (canonical or legacy).

        Canonical: has 'regions' key.
        Legacy: has 'grid' key at top level — auto-converted via _to_canonical().

        Required: at least one region with 'wiring'.
        Optional: regions with 'source', 'noise', 'spiking', 'daemon'; connections[].
        """
        config = _to_canonical(config)
        self._config = config
        self.generation = 0

        # Find tissue region (first with wiring) and optional input source region
        tissue_cfg = next((r for r in config["regions"] if "wiring" in r), None)
        if tissue_cfg is None:
            raise ValueError("config must have at least one region with 'wiring'")

        source_region = next((r for r in config["regions"] if "source" in r), None)
        connections = config.get("connections", [])
        input_conn = connections[0] if connections else None

        # ── Grid ──
        grid = tissue_cfg["grid"]
        self.width = int(grid.get("width", 50))
        self.height = int(grid.get("height", 50))

        # ── Wiring ──
        wiring = tissue_cfg["wiring"]
        mask_id: str = wiring.get("mask", "")
        self.process_mode = wiring.get("process_mode", "min_vs_max")

        self.dendrite_exc_weight = wiring.get("dendrite_exc_weight")
        self.dendrite_inh_weight = wiring.get("dendrite_inh_weight")

        tf = wiring.get("tension_function")
        if tf and isinstance(tf, dict):
            self._tension_fns = [(k, float(v)) for k, v in tf.items()]
        else:
            self._tension_fns = []

        # ── Learning ──
        # wiring.learning_rate → internal tissue dendrites (exc + inh)
        # connections[0].learning.rate → inter-region dendrites (input)
        # connections[0].learning.exclude_range → dead zone for input weights
        wiring_lr = wiring.get("learning_rate")

        conn_learning = input_conn.get("learning") if input_conn else None
        if isinstance(conn_learning, dict):
            conn_lr: float | None = float(conn_learning.get("rate", 0.0))
            er = conn_learning.get("exclude_range")
            self._conn_exclude_range = (float(er[0]), float(er[1])) if er else None
        else:
            # backward compat: bare learning_rate on connection
            raw = input_conn.get("learning_rate") if input_conn else None
            conn_lr = float(raw) if raw is not None else None
            self._conn_exclude_range = None

        self.learning_rate = 1.0
        self.lr_exc = float(wiring_lr) if wiring_lr is not None else 0.0
        self.lr_inh = float(wiring_lr) if wiring_lr is not None else 0.0
        self.lr_input = float(conn_lr) if conn_lr is not None else 0.0
        self.learning_enabled = self.lr_exc > 0 or self.lr_inh > 0 or self.lr_input > 0

        # ── Input (opt-in) ──
        self.input_enabled = source_region is not None

        if source_region is not None:
            source = source_region["source"]
            input_grid = source_region["grid"]
            self.input_resolution = int(input_grid.get("width", 20))
            self.input_text = source.get("text", "") or ""
            self.frames_per_char = max(1, int(source.get("frames_per_char", 10)))
            self._font_id = source.get("font", "press_start_2p")
            self._font_size = int(source.get("font_size", 10))

            if input_conn is not None:
                self.dendrite_input_weight = float(input_conn.get("weight", 0.2))
                self.input_density = float(input_conn.get("density", 1.0))
                if input_conn.get("type") == "portion":
                    pr = input_conn["portion"]
                    self.input_portion = (int(pr[0]), int(pr[1]))
                else:
                    self.input_portion = None
            else:
                self.dendrite_input_weight = 0.2
                self.input_density = 1.0
                self.input_portion = None
        else:
            self.input_text = ""
            self.input_resolution = 0
            self.input_density = 1.0
            self.input_portion = None

        # ── Noise (opt-in) — lives in source because it affects input rendering ──
        noise_cfg = source_region["source"].get("noise") if source_region else None
        if noise_cfg:
            self.background_noise = float(noise_cfg.get("background", 0.0))
            self.shift_noise = bool(noise_cfg.get("shift", False))
            self.inter_char_noise = bool(noise_cfg.get("inter_char", False))
        else:
            self.background_noise = 0.0
            self.shift_noise = False
            self.inter_char_noise = False

        # ── Spiking (opt-in) ──
        spiking_cfg = tissue_cfg.get("spiking")
        self.adaptation_enabled = spiking_cfg is not None
        if spiking_cfg:
            self.up_ticks = int(spiking_cfg.get("up_ticks", 5))
            self.down_ticks = int(spiking_cfg.get("down_ticks", 5))
        else:
            self.up_ticks = 5
            self.down_ticks = 5

        # ── Daemon detection config (opt-in) ──
        daemon_cfg = tissue_cfg.get("daemon")
        if daemon_cfg:
            self._daemon_threshold = float(daemon_cfg.get("threshold", _DAEMON_THRESHOLD))
            self._min_daemon_size = int(daemon_cfg.get("min_size", _MIN_DAEMON_SIZE))
        else:
            self._daemon_threshold = _DAEMON_THRESHOLD
            self._min_daemon_size = _MIN_DAEMON_SIZE

        # ── Mask setup ──
        if "deamon" in wiring:
            self._mask_type = "kohonen"
            deamon_cfg = wiring["deamon"]
            self._random_weights = not deamon_cfg.get("fixed", False)
            raw_mask = compile_deamon_wiring(deamon_cfg)
            centroid_cfg = deamon_cfg.get("centroid") or {}
            self._centroid_jitter = 1 if centroid_cfg.get("random") else 0
        else:
            self._mask_type = get_mask_type(mask_id)
            self._random_weights = get_random_weights(mask_id)
            raw_mask = get_mask(mask_id)
            self._centroid_jitter = 0

        # Apply flat dendrite-weight override when specified in wiring config.
        has_override = (
            self.dendrite_exc_weight is not None or self.dendrite_inh_weight is not None
        )
        if has_override:
            mask = []
            for d in raw_mask:
                peso = d["peso_dendrita"]
                if self.dendrite_exc_weight is not None and peso > 0:
                    mask.append({**d, "peso_dendrita": self.dendrite_exc_weight})
                elif self.dendrite_inh_weight is not None and peso < 0:
                    mask.append({**d, "peso_dendrita": self.dendrite_inh_weight})
                else:
                    mask.append(d)
        else:
            mask = raw_mask

        # ── Neurons ──
        is_wolfram = self._mask_type == "wolfram"
        n_input = self.input_resolution * self.input_resolution if self.input_enabled else 0
        self._input_start_idx = self.width * self.height

        if is_wolfram:
            constructor = Constructor()
            self.brain, self.regiones = constructor.crear_grilla(
                width=self.width,
                height=self.height,
                filas_entrada=[self.height - 1],
                filas_salida=[],
                umbral=0.99,
            )
        else:
            tissue_neurons: list[Neurona] = []
            for y in range(self.height):
                for x in range(self.width):
                    tissue_neurons.append(
                        Neurona(id=Constructor.key_by_coord(x, y), umbral=0.0)
                    )

            input_neurons: list[Neurona] = []
            if self.input_enabled:
                for idx in range(n_input):
                    input_neurons.append(NeuronaEntrada(id=f"inp_{idx}"))

            all_neurons: list[Neurona] = tissue_neurons + input_neurons
            self.brain = Brain(neuronas=all_neurons)

            region_tissue = Region(nombre="tissue")
            for n_obj in tissue_neurons:
                region_tissue.agregar(n_obj)
            self.regiones = {"tissue": region_tissue}

            if self.input_enabled:
                region_input = Region(nombre="input")
                for n_obj in input_neurons:
                    region_input.agregar(n_obj)
                self.regiones["input"] = region_input

        # ── Wiring ──
        constructor = Constructor()
        constructor.aplicar_mascara_2d(
            self.brain,
            self.width,
            self.height,
            mask,
            random_weights=self._random_weights,
            centroid_jitter=self._centroid_jitter,
        )

        # ── Input dendrites ──
        if self.input_enabled and not is_wolfram:
            input_neuron_list = list(self.regiones["input"].neuronas.values())
            tissue_list = list(self.regiones["tissue"].neuronas.values())

            if self.input_portion is not None:
                # Divide both the tissue grid and the input grid into
                # n_div_y × n_div_x equal regions. Each tissue neuron in
                # region (ry, rx) connects fully to every input neuron in the
                # corresponding input region (ry, rx).
                n_div_y, n_div_x = self.input_portion
                res = self.input_resolution
                inp_by_id = {n.id: n for n in input_neuron_list}

                inp_regions: dict[tuple[int, int], list] = {}
                for ry in range(n_div_y):
                    for rx in range(n_div_x):
                        py0 = ry * res // n_div_y
                        py1 = (ry + 1) * res // n_div_y
                        px0 = rx * res // n_div_x
                        px1 = (rx + 1) * res // n_div_x
                        inp_regions[(ry, rx)] = [
                            inp_by_id[f"inp_{py * res + px}"]
                            for py in range(py0, py1)
                            for px in range(px0, px1)
                        ]

                for tissue_n in tissue_list:
                    x_str, y_str = tissue_n.id[1:].split("y")
                    tx, ty = int(x_str), int(y_str)
                    rx = min(tx * n_div_x // self.width, n_div_x - 1)
                    ry = min(ty * n_div_y // self.height, n_div_y - 1)
                    sinapsis_list: list[Sinapsis] = [
                        Sinapsis(neurona_entrante=inp_n, peso=random.uniform(0.2, 1.0))
                        for inp_n in inp_regions[(ry, rx)]
                    ]
                    tissue_n.dendritas.append(
                        Dendrita(sinapsis=sinapsis_list, peso=self.dendrite_input_weight)
                    )
            else:
                k = max(1, round(len(input_neuron_list) * self.input_density))
                for tissue_n in tissue_list:
                    sampled = random.sample(input_neuron_list, k) if k < len(input_neuron_list) else input_neuron_list
                    sinapsis_list = []
                    for inp_n in sampled:
                        sinapsis_list.append(
                            Sinapsis(neurona_entrante=inp_n, peso=random.uniform(0.2, 1.0))
                        )
                    tissue_n.dendritas.append(
                        Dendrita(sinapsis=sinapsis_list, peso=self.dendrite_input_weight)
                    )

        # ── Initialization ──
        if is_wolfram:
            for neurona in self.brain.neuronas:
                neurona.activar_external(0.0)
            center_x = self.width // 2
            bottom_y = self.height - 1
            self.brain.get_neurona(
                f"x{center_x}y{bottom_y}"
            ).activar_external(1.0)
        else:
            tissue_region = self.regiones.get("tissue")
            if tissue_region:
                for neurona in tissue_region.neuronas.values():
                    neurona.activar_external(random.random())

        # ── Compile ──
        self.brain_tensor = ConstructorTensor.compilar(
            self.brain,
            max_active_steps=self.up_ticks,
            refractory_steps=self.down_ticks,
            adaptation_enabled=self.adaptation_enabled,
            process_mode=self.process_mode,
            tension_fns=self._tension_fns,
        )

        # ── Pre-render characters ──
        self._char_images = {}
        if self.input_enabled and not self._is_synthetic_input():
            for char in set(self.input_text):
                self._char_images[char] = render_char(
                    char, self.input_resolution,
                    font_id=self._font_id, font_size=self._font_size,
                )

        # ── Frame tracking ──
        self._char_index = 0
        self._frame_in_char = 0
        self._in_gap = False
        self._rng = np.random.default_rng()

        # ── Daemon stats ──
        self._daemon_history.clear()
        self._last_history_gen = -1

        # ── Project initial frame ──
        if self.input_enabled:
            self._generate_and_project()

    # ── Input helpers ──

    def _is_synthetic_input(self) -> bool:
        if not self.input_text:
            return False
        return all(
            tok.strip() in self._SYNTHETIC_PATTERNS
            for tok in self.input_text.split(",")
        )

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
                y0 = max(0, y - 1)
                y1 = min(res, y + 2)
                frame[y0:y1, :] = 1.0
        elif name == "BARS_V":
            for i in range(3):
                x = int(res * (i + 1) / 4)
                x0 = max(0, x - 1)
                x1 = min(res, x + 2)
                frame[:, x0:x1] = 1.0
        elif name == "DOT_TL":
            frame[:5, :5] = 1.0
        elif name == "DOT_BR":
            frame[-5:, -5:] = 1.0
        return frame

    def _generate_and_project(self) -> None:
        res = self.input_resolution

        if not self.input_text:
            frame = self._rng.integers(0, 2, size=(res, res)).astype(np.float64)
        elif self._in_gap:
            frame = self._rng.integers(0, 2, size=(res, res)).astype(np.float64)
        elif self._is_synthetic_input():
            tokens = [t.strip() for t in self.input_text.split(",")]
            pattern_name = tokens[self._char_index % len(tokens)]
            frame = self._make_synthetic(pattern_name, res)
            if self.background_noise > 0:
                frame = apply_white_noise(frame, noise_prob=self.background_noise, rng=self._rng)
        else:
            char = self.input_text[self._char_index]
            base = self._char_images[char]
            frame = base.copy()
            if self.shift_noise:
                frame = apply_shift_noise(frame, self._rng)
            if self.background_noise > 0:
                frame = apply_white_noise(frame, noise_prob=self.background_noise, rng=self._rng)

        self._current_input_frame = frame

        if self.brain_tensor is not None:
            flat = torch.from_numpy(frame.flatten()).float()
            start = self._input_start_idx
            end = start + len(flat)
            self.brain_tensor.valores[start:end] = flat

    # ── Processing ──

    def step(self) -> dict[str, Any]:
        if self.input_enabled:
            self._generate_and_project()
        self.brain_tensor.procesar()

        if self.learning_enabled and self.brain_tensor is not None:
            self.brain_tensor.learn(
                lr=self.learning_rate,
                lr_exc=self.lr_exc,
                lr_inh=self.lr_inh,
                lr_input=self.lr_input,
                conn_exclude_range=self._conn_exclude_range,
            )

        self.generation += 1

        if self.input_enabled and self.input_text:
            n_items = (
                len([t.strip() for t in self.input_text.split(",")])
                if self._is_synthetic_input()
                else len(self.input_text)
            )
            self._frame_in_char += 1
            if self._in_gap:
                if self._frame_in_char >= self.frames_per_char:
                    self._in_gap = False
                    self._frame_in_char = 0
                    self._char_index = (self._char_index + 1) % n_items
            elif self._frame_in_char >= self.frames_per_char:
                if self.inter_char_noise:
                    self._in_gap = True
                    self._frame_in_char = 0
                else:
                    self._frame_in_char = 0
                    self._char_index = (self._char_index + 1) % n_items

        return {
            "type": "frame",
            "generation": self.generation,
            "grid": self.get_frame(),
            "stats": self.get_stats(),
        }

    def step_n(self, count: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for _ in range(count):
            result = self.step()
        return result

    def click(self, x: int, y: int) -> None:
        if self.brain_tensor is None:
            return
        idx = y * self.width + x
        limit = self._input_start_idx if self.input_enabled else self.brain_tensor.n_real
        if 0 <= idx < limit:
            current = self.brain_tensor.valores[idx].item()
            self.brain_tensor.set_valor(idx, 0.0 if current >= 0.5 else 1.0)

    def get_frame(self) -> list[list[float]]:
        if self.brain_tensor:
            return self.brain_tensor.get_grid(self.width, self.height)
        return super().get_frame()

    def get_tension_frame(self) -> list[list[float]] | None:
        if self.brain_tensor:
            return self.brain_tensor.get_tension_grid(self.width, self.height)
        return None

    def get_input_frame(self) -> list[list[float]] | None:
        if not self.input_enabled:
            return None
        if self._current_input_frame is not None:
            return self._current_input_frame.tolist()
        return None

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

        if self.input_enabled:
            if not self.input_text:
                current_char = ""
            elif self._in_gap:
                current_char = "gap"
            elif self._is_synthetic_input():
                tokens = [t.strip() for t in self.input_text.split(",")]
                current_char = tokens[self._char_index % len(tokens)]
            else:
                current_char = self.input_text[self._char_index]

            stats.update({
                "current_char": current_char,
                "char_index": self._char_index,
                "frame_in_char": self._frame_in_char,
                "frames_per_char": self.frames_per_char,
                "input_resolution": self.input_resolution,
            })

        return stats

    def inspect(self, x: int, y: int) -> dict[str, Any]:
        """Return connection weights from brain_tensor (live, trained weights)."""
        if self.brain_tensor is None:
            result = super().inspect(x, y)
            result["input_weight_grid"] = None
            return result

        neuron_idx = y * self.width + x
        n_input = (self.input_resolution * self.input_resolution) if self.input_enabled else 0
        input_start = self._input_start_idx
        input_end = input_start + n_input

        sources = self.brain_tensor.indices_fuente[neuron_idx]
        weights = self.brain_tensor.pesos_sinapsis[neuron_idx]
        dend_weights = self.brain_tensor.pesos_dendrita[neuron_idx]
        valid = self.brain_tensor.mascara_valida[neuron_idx]
        dend_ids = self.brain_tensor.dendrita_ids[neuron_idx]

        total_sinapsis = int(valid.sum().item())
        total_dendritas = int(dend_ids[valid].unique().numel()) if total_sinapsis > 0 else 0

        tissue_pesos: dict[int, float] = {}
        input_weights: list[float] = [0.0] * n_input

        for i in range(sources.shape[0]):
            if not valid[i]:
                continue
            src = sources[i].item()
            w = weights[i].item()
            dw = dend_weights[i].item()

            if self.input_enabled and input_start <= src < input_end:
                input_weights[src - input_start] = w
            elif src < input_start:
                effective = w * dw
                tissue_pesos[src] = tissue_pesos.get(src, 0.0) + effective

        for src in tissue_pesos:
            tissue_pesos[src] = max(-1.0, min(1.0, tissue_pesos[src]))

        weight_grid: list[list[float | None]] = []
        for row in range(self.height):
            fila: list[float | None] = []
            for col in range(self.width):
                if col == x and row == y:
                    fila.append(999)
                else:
                    idx = row * self.width + col
                    if idx in tissue_pesos:
                        fila.append(tissue_pesos[idx])
                    else:
                        fila.append(None)
            weight_grid.append(fila)

        activation = self.brain_tensor.valores[neuron_idx].item()
        tension = self.brain_tensor.tensiones[neuron_idx].item()

        result: dict[str, Any] = {
            "type": "connections",
            "x": x,
            "y": y,
            "activation": round(activation, 4),
            "tension": round(tension, 4),
            "total_dendritas": total_dendritas,
            "total_sinapsis": total_sinapsis,
            "weight_grid": weight_grid,
        }

        if self.input_enabled:
            res = self.input_resolution
            input_grid: list[list[float]] = []
            for r in range(res):
                input_grid.append(input_weights[r * res : (r + 1) * res])
            result["input_weight_grid"] = input_grid
            result["input_weight_width"] = res
            result["input_weight_height"] = res
        else:
            result["input_weight_grid"] = None

        return result

    def update_config(self, config: dict[str, Any]) -> bool:
        """Apply config changes to a running experiment.

        Accepts both canonical and legacy format — converts via _to_canonical().
        Returns True if only soft updates were applied (no rebuild needed).
        """
        if self.brain_tensor is None:
            return False

        new_config = _to_canonical(config)
        old_config = self._config  # always canonical after setup()

        old_tissue = next((r for r in old_config.get("regions", []) if "wiring" in r), {})
        new_tissue = next((r for r in new_config.get("regions", []) if "wiring" in r), {})
        old_connections = old_config.get("connections", [])
        new_connections = new_config.get("connections", [])
        old_conn = old_connections[0] if old_connections else {}
        new_conn = new_connections[0] if new_connections else {}

        needs_reconnect = False

        if new_tissue.get("grid") != old_tissue.get("grid"):
            needs_reconnect = True

        if not needs_reconnect:
            old_wiring = old_tissue.get("wiring", {})
            new_wiring = new_tissue.get("wiring", {})
            for k in ("mask", "dendrite_exc_weight", "dendrite_inh_weight"):
                if new_wiring.get(k) != old_wiring.get(k):
                    needs_reconnect = True
                    break

        if not needs_reconnect:
            old_has_source = any("source" in r for r in old_config.get("regions", []))
            new_has_source = any("source" in r for r in new_config.get("regions", []))
            if old_has_source != new_has_source:
                needs_reconnect = True
            elif new_has_source:
                old_src = next((r for r in old_config["regions"] if "source" in r), {})
                new_src = next((r for r in new_config["regions"] if "source" in r), {})
                if old_src.get("grid") != new_src.get("grid"):
                    needs_reconnect = True
                elif old_conn.get("weight") != new_conn.get("weight"):
                    needs_reconnect = True

        if needs_reconnect:
            self.setup(new_config)
            return False

        # ── Soft updates ──
        new_wiring = new_tissue.get("wiring", {})

        if "process_mode" in new_wiring:
            self.process_mode = new_wiring["process_mode"]
            if self.brain_tensor is not None:
                self.brain_tensor.process_mode = self.process_mode

        if "tension_function" in new_wiring:
            tf = new_wiring["tension_function"]
            if tf and isinstance(tf, dict):
                self._tension_fns = [(k, float(v)) for k, v in tf.items()]
            else:
                self._tension_fns = []
            if self.brain_tensor is not None:
                self.brain_tensor.tension_fns = self._tension_fns

        wiring_lr = new_wiring.get("learning_rate")
        conn_lr = new_conn.get("learning_rate")
        self.learning_rate = 1.0
        self.lr_exc = float(wiring_lr) if wiring_lr is not None else 0.0
        self.lr_inh = float(wiring_lr) if wiring_lr is not None else 0.0
        self.lr_input = float(conn_lr) if conn_lr is not None else 0.0
        self.learning_enabled = self.lr_exc > 0 or self.lr_inh > 0 or self.lr_input > 0

        new_noise = new_tissue.get("noise")
        if new_noise:
            self.background_noise = float(new_noise.get("background", 0.0))
            self.shift_noise = bool(new_noise.get("shift", False))
            self.inter_char_noise = bool(new_noise.get("inter_char", False))
        else:
            self.background_noise = 0.0
            self.shift_noise = False
            self.inter_char_noise = False

        new_spiking = new_tissue.get("spiking")
        if new_spiking:
            self.adaptation_enabled = True
            self.up_ticks = int(new_spiking.get("up_ticks", self.up_ticks))
            self.down_ticks = int(new_spiking.get("down_ticks", self.down_ticks))
            if self.brain_tensor is not None:
                self.brain_tensor.adaptation_enabled = True
                self.brain_tensor.max_active_steps = self.up_ticks
                self.brain_tensor.refractory_steps = self.down_ticks
        else:
            self.adaptation_enabled = False
            if self.brain_tensor is not None:
                self.brain_tensor.adaptation_enabled = False

        if self.input_enabled:
            new_src = next((r for r in new_config.get("regions", []) if "source" in r), None)
            if new_src:
                source = new_src["source"]

                font_changed = False
                if "font" in source and source["font"] != self._font_id:
                    self._font_id = source["font"]
                    font_changed = True
                if "font_size" in source and source["font_size"] != self._font_size:
                    self._font_size = int(source["font_size"])
                    font_changed = True

                text_changed = False
                if "text" in source:
                    new_text = source["text"] or ""
                    if new_text != self.input_text:
                        self.input_text = new_text
                        text_changed = True
                        self._char_index = 0
                        self._frame_in_char = 0
                        self._in_gap = False

                if "frames_per_char" in source:
                    self.frames_per_char = max(1, int(source["frames_per_char"]))

                if (font_changed or text_changed) and not self._is_synthetic_input():
                    self._char_images = {}
                    for char in set(self.input_text):
                        self._char_images[char] = render_char(
                            char, self.input_resolution,
                            font_id=self._font_id, font_size=self._font_size,
                        )

        self._config = new_config
        return True

    def reset(self) -> None:
        self.setup(self._config)

    def is_complete(self) -> bool:
        return False

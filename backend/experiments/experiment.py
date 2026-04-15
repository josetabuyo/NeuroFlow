"""Unified NeuroFlow experiment.

Supports all features through opt-in config sections:
  - grid + wiring (required)
  - input (optional: ASCII/synthetic input stream)
  - noise (optional: background, shift, inter-char)
  - learning (optional: Hebbian weight updates)
  - spiking (optional: spike frequency adaptation)

Config is nested JSON. Section present = feature enabled.
Section absent = feature disabled.
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
from core.masks import get_mask, get_mask_type, get_random_weights
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


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate config, warn on missing required fields, fill safe defaults.

    Required fields get warnings + defaults.
    Optional feature sections are never added — absent means disabled.
    """
    config = {**config}

    grid = config.get("grid")
    if grid is None:
        logger.warning("config missing 'grid' section, defaulting to {width: 50, height: 50}")
        config["grid"] = {"width": 50, "height": 50}
    else:
        grid = {**grid}
        if "width" not in grid:
            logger.warning("grid.width missing, defaulting to 50")
            grid["width"] = 50
        if "height" not in grid:
            logger.warning("grid.height missing, defaulting to 50")
            grid["height"] = 50
        config["grid"] = grid

    wiring = config.get("wiring")
    if wiring is None:
        logger.warning("config missing 'wiring' section, defaulting to {mask: 'deamon_3_en_50', process_mode: 'min_vs_max'}")
        config["wiring"] = {"mask": "deamon_3_en_50", "process_mode": "min_vs_max"}
    else:
        wiring = {**wiring}
        if "mask" not in wiring:
            logger.warning("wiring.mask missing, defaulting to 'deamon_3_en_50'")
            wiring["mask"] = "deamon_3_en_50"
        if "process_mode" not in wiring:
            logger.warning("wiring.process_mode missing, defaulting to 'min_vs_max'")
            wiring["process_mode"] = "min_vs_max"
        config["wiring"] = wiring

    return config


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
        self.learning_enabled: bool = False
        self.learning_rate: float = 0.01
        self.lr_exc: float = 1.0
        self.lr_inh: float = 1.0
        self.lr_input: float = 1.0

        # Spiking state
        self.adaptation_enabled: bool = False
        self.up_ticks: int = 5
        self.down_ticks: int = 5

        # Daemon stats
        self._daemon_history: deque[int] = deque(maxlen=_STABILITY_WINDOW)
        self._last_history_gen: int = -1

    def setup(self, config: dict[str, Any]) -> None:
        """Build the network from a nested config.

        Required sections: grid, wiring (auto-defaulted with warnings).
        Optional sections: input, noise, learning, spiking.
        """
        config = _validate_config(config)
        self._config = config
        self.generation = 0

        # ── Grid ──
        grid = config["grid"]
        self.width = grid["width"]
        self.height = grid["height"]

        # ── Wiring ──
        wiring = config["wiring"]
        mask_id: str = wiring["mask"]
        self.process_mode = wiring["process_mode"]

        self.dendrite_exc_weight = wiring.get("dendrite_exc_weight")
        self.dendrite_inh_weight = wiring.get("dendrite_inh_weight")

        tf = wiring.get("tension_function")
        if tf and isinstance(tf, dict):
            self._tension_fns = [(k, float(v)) for k, v in tf.items()]
        else:
            self._tension_fns = []

        # ── Input (opt-in) ──
        input_cfg = config.get("input")
        self.input_enabled = input_cfg is not None

        if input_cfg:
            self.input_text = input_cfg.get("text", "") or ""
            self.input_resolution = input_cfg.get("resolution", 20)
            self.frames_per_char = max(1, input_cfg.get("frames_per_char", 10))
            self.dendrite_input_weight = input_cfg.get("dendrite_input_weight", 0.2)
            self.input_density = float(input_cfg.get("density", 1.0))
            self._font_id = input_cfg.get("font", "press_start_2p")
            self._font_size = input_cfg.get("font_size", 10)
        else:
            self.input_text = ""
            self.input_resolution = 0
            self.input_density = 1.0

        # ── Noise (opt-in) ──
        noise_cfg = config.get("noise")
        if noise_cfg:
            self.background_noise = float(noise_cfg.get("background", 0.0))
            self.shift_noise = noise_cfg.get("shift", False)
            self.inter_char_noise = noise_cfg.get("inter_char", False)
        else:
            self.background_noise = 0.0
            self.shift_noise = False
            self.inter_char_noise = False

        # ── Learning (opt-in) ──
        learning_cfg = config.get("learning")
        self.learning_enabled = learning_cfg is not None
        if learning_cfg:
            self.learning_rate = learning_cfg.get("rate", 1.0)
            self.lr_exc   = learning_cfg.get("lr_exc",   1.0)
            self.lr_inh   = learning_cfg.get("lr_inh",   1.0)
            self.lr_input = learning_cfg.get("lr_input", 1.0)
        else:
            self.lr_exc = self.lr_inh = self.lr_input = 1.0

        # ── Spiking (opt-in) ──
        spiking_cfg = config.get("spiking")
        self.adaptation_enabled = spiking_cfg is not None
        if spiking_cfg:
            self.up_ticks = spiking_cfg.get("up_ticks", 5)
            self.down_ticks = spiking_cfg.get("down_ticks", 5)
        else:
            self.up_ticks = 5
            self.down_ticks = 5

        # ── Mask setup ──
        self._mask_type = get_mask_type(mask_id)
        self._random_weights = get_random_weights(mask_id)
        raw_mask = get_mask(mask_id)

        mask = []
        for d in raw_mask:
            peso = d["peso_dendrita"]
            if self.dendrite_exc_weight is not None and peso > 0:
                mask.append({**d, "peso_dendrita": self.dendrite_exc_weight})
            elif self.dendrite_inh_weight is not None and peso < 0:
                mask.append({**d, "peso_dendrita": self.dendrite_inh_weight})
            else:
                mask.append(d)

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
        )

        # ── Input dendrites ──
        if self.input_enabled and not is_wolfram:
            input_neuron_list = list(self.regiones["input"].neuronas.values())
            tissue_list = list(self.regiones["tissue"].neuronas.values())
            k = max(1, round(len(input_neuron_list) * self.input_density))
            for tissue_n in tissue_list:
                sampled = random.sample(input_neuron_list, k) if k < len(input_neuron_list) else input_neuron_list
                sinapsis_list: list[Sinapsis] = []
                for inp_n in sampled:
                    peso = random.uniform(0.2, 1.0)
                    sinapsis_list.append(
                        Sinapsis(neurona_entrante=inp_n, peso=peso)
                    )
                dendrita = Dendrita(
                    sinapsis=sinapsis_list, peso=self.dendrite_input_weight
                )
                tissue_n.dendritas.append(dendrita)

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
        active = int((vals > _DAEMON_THRESHOLD).sum().item())

        # Daemon detection
        count, daemon_indices, noise_indices, sizes = _detect_daemons(
            self.brain_tensor.valores, self.width, self.height, _DAEMON_THRESHOLD
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

        Returns True if only soft updates were applied (no rebuild needed).
        """
        if self.brain_tensor is None:
            return False

        hard_keys = {"grid", "wiring"}
        hard_input_keys = {"resolution", "dendrite_input_weight"}

        needs_reconnect = False
        if "grid" in config:
            old_grid = self._config.get("grid", {})
            new_grid = config["grid"]
            if new_grid.get("width") != old_grid.get("width") or new_grid.get("height") != old_grid.get("height"):
                needs_reconnect = True

        if "wiring" in config and not needs_reconnect:
            old_wiring = self._config.get("wiring", {})
            new_wiring = config["wiring"]
            for k in ("mask", "dendrite_exc_weight", "dendrite_inh_weight"):
                if new_wiring.get(k) != old_wiring.get(k):
                    needs_reconnect = True
                    break

        if "input" in config and not needs_reconnect:
            new_input = config.get("input") or {}
            old_input = self._config.get("input") or {}
            for k in hard_input_keys:
                if new_input.get(k) != old_input.get(k):
                    needs_reconnect = True
                    break
            input_was_enabled = self._config.get("input") is not None
            input_now_enabled = config.get("input") is not None
            if input_was_enabled != input_now_enabled:
                needs_reconnect = True

        if needs_reconnect:
            self.setup(config)
            return False

        # Soft updates
        if "learning" in config:
            learning_cfg = config["learning"]
            if learning_cfg:
                self.learning_enabled = True
                self.learning_rate = learning_cfg.get("rate", self.learning_rate)
                self.lr_exc   = learning_cfg.get("lr_exc",   self.lr_exc)
                self.lr_inh   = learning_cfg.get("lr_inh",   self.lr_inh)
                self.lr_input = learning_cfg.get("lr_input", self.lr_input)
            else:
                self.learning_enabled = False
        elif "learning" not in config and self._config.get("learning") is not None and config.get("learning") is None:
            self.learning_enabled = False

        if "noise" in config:
            noise_cfg = config["noise"]
            if noise_cfg:
                self.background_noise = float(noise_cfg.get("background", 0.0))
                self.shift_noise = noise_cfg.get("shift", False)
                self.inter_char_noise = noise_cfg.get("inter_char", False)
            else:
                self.background_noise = 0.0
                self.shift_noise = False
                self.inter_char_noise = False

        if "spiking" in config:
            spiking_cfg = config["spiking"]
            if spiking_cfg:
                self.adaptation_enabled = True
                self.up_ticks = spiking_cfg.get("up_ticks", self.up_ticks)
                self.down_ticks = spiking_cfg.get("down_ticks", self.down_ticks)
                if self.brain_tensor is not None:
                    self.brain_tensor.adaptation_enabled = True
                    self.brain_tensor.max_active_steps = self.up_ticks
                    self.brain_tensor.refractory_steps = self.down_ticks
            else:
                self.adaptation_enabled = False
                if self.brain_tensor is not None:
                    self.brain_tensor.adaptation_enabled = False

        wiring_cfg = config.get("wiring")
        if wiring_cfg:
            if "process_mode" in wiring_cfg:
                self.process_mode = wiring_cfg["process_mode"]
                if self.brain_tensor is not None:
                    self.brain_tensor.process_mode = self.process_mode
            if "tension_function" in wiring_cfg:
                tf = wiring_cfg["tension_function"]
                if tf and isinstance(tf, dict):
                    self._tension_fns = [(k, float(v)) for k, v in tf.items()]
                else:
                    self._tension_fns = []
                if self.brain_tensor is not None:
                    self.brain_tensor.tension_fns = self._tension_fns

        input_cfg = config.get("input")
        if input_cfg and self.input_enabled:
            font_changed = False
            if "font" in input_cfg and input_cfg["font"] != self._font_id:
                self._font_id = input_cfg["font"]
                font_changed = True
            if "font_size" in input_cfg and input_cfg["font_size"] != self._font_size:
                self._font_size = input_cfg["font_size"]
                font_changed = True

            text_changed = False
            if "text" in input_cfg:
                new_text = input_cfg["text"] or ""
                if new_text != self.input_text:
                    self.input_text = new_text
                    text_changed = True
                    self._char_index = 0
                    self._frame_in_char = 0
                    self._in_gap = False

            if "frames_per_char" in input_cfg:
                self.frames_per_char = max(1, input_cfg["frames_per_char"])

            if (font_changed or text_changed) and not self._is_synthetic_input():
                self._char_images = {}
                for char in set(self.input_text):
                    self._char_images[char] = render_char(
                        char, self.input_resolution,
                        font_id=self._font_id, font_size=self._font_size,
                    )

        self._config = config
        return True

    def reset(self) -> None:
        self.setup(self._config)

    def is_complete(self) -> bool:
        return False

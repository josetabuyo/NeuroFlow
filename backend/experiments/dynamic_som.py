"""Dynamic SOM experiment — self-organizing map with visual input streams.

Projects a video stream (ASCII characters with configurable noise) into an
input region. Each neuron in the tissue gets a positive dendrite connected
to all input neurons, plus local connectivity from a chosen wiring mask.

The tissue processes patterns through its internal connections (daemons)
while receiving external input through the input dendrite.
"""

from __future__ import annotations

import random
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


class DynamicSOMExperiment(Experimento):
    """Dynamic SOM: tissue + input region with visual stream projection."""

    def __init__(self) -> None:
        super().__init__()
        self._config: dict[str, Any] = {}
        self.brain_tensor = None
        self.input_resolution: int = 20
        self.input_text: str = "AB"
        self.frames_per_char: int = 10
        self.input_dendrite_weight: float = 0.2
        self.deamon_exc_weight: float = 0.5
        self.deamon_inh_weight: float = -0.5
        self.white_noise_enabled: bool = True
        self.noise_prob: float = 0.05
        self.shift_noise_enabled: bool = False
        self.inter_char_noise: bool = True
        self.learning_enabled: bool = True
        self.learning_rate: float = 0.01
        self.adaptation_enabled: bool = True
        self.max_active_steps: int = 5
        self.refractory_steps: int = 5
        self._font_id: str = "press_start_2p"
        self._font_size: int = 10
        self._char_images: dict[str, np.ndarray] = {}
        self._char_index: int = 0
        self._frame_in_char: int = 0
        self._in_gap: bool = False
        self._current_input_frame: np.ndarray | None = None
        self._input_start_idx: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

    def setup(self, config: dict[str, Any]) -> None:
        """Build tissue grid + input region, apply mask, add input dendrites.

        Config keys:
            width (int): Tissue width (default 50).
            height (int): Tissue height (default 50).
            mask (str): Wiring preset ID (default "deamon_3_en_50").
            input_text (str): Characters to cycle through (default "AB").
                Empty string means pure white noise with no character rendering.
            input_resolution (int): Input image size, square (default 20).
            frames_per_char (int): Frames per character (default 10).
            input_dendrite_weight (float): Weight of input dendrite (default 0.2).
            deamon_exc_weight (float): Weight for excitatory dendrites in mask (default 0.5).
            deamon_inh_weight (float): Weight for inhibitory dendrites in mask (default -0.5).
            white_noise (bool): Enable white noise (default True).
            shift_noise (bool): Enable shift noise (default False).
            inter_char_noise (bool): White noise between characters (default True).
                When False, a black image is projected between characters.
            font (str): Font ID from the registry (default "press_start_2p").
            font_size (int): Font size in target-res pixels (default 10).
        """
        self._config = config
        self.width = config.get("width", 50)
        self.height = config.get("height", 50)
        self.generation = 0

        mask_id: str = config.get("mask", "deamon_3_en_50")
        self.input_text = config.get("input_text", "AB") or ""
        self.input_resolution = config.get("input_resolution", 20)
        self.frames_per_char = max(1, config.get("frames_per_char", 10))
        self.input_dendrite_weight = config.get("input_dendrite_weight", 0.2)
        self.deamon_exc_weight = config.get("deamon_exc_weight", 0.5)
        self.deamon_inh_weight = config.get("deamon_inh_weight", -0.5)
        self.white_noise_enabled = config.get("white_noise", True)
        self.noise_prob = config.get("noise_prob", 0.05)
        self.shift_noise_enabled = config.get("shift_noise", False)
        self.inter_char_noise = config.get("inter_char_noise", True)
        self.learning_enabled = config.get("learning", True)
        self.learning_rate = config.get("learning_rate", 0.01)
        self.adaptation_enabled = config.get("spike_adaptation", False)
        self.max_active_steps = config.get("max_active_steps", 5)
        self.refractory_steps = config.get("refractory_steps", 5)
        self._font_id = config.get("font", "press_start_2p")
        self._font_size = config.get("font_size", 10)

        self._mask_type = get_mask_type(mask_id)
        self._random_weights = get_random_weights(mask_id)
        raw_mask = get_mask(mask_id)

        # Override mask dendrite weights with configurable values
        mask = []
        for d in raw_mask:
            peso = d["peso_dendrita"]
            if peso > 0:
                mask.append({**d, "peso_dendrita": self.deamon_exc_weight})
            elif peso < 0:
                mask.append({**d, "peso_dendrita": self.deamon_inh_weight})
            else:
                mask.append(d)

        n_input = self.input_resolution * self.input_resolution
        self._input_start_idx = self.width * self.height

        # --- Create neurons ---
        tissue_neurons: list[Neurona] = []
        for y in range(self.height):
            for x in range(self.width):
                tissue_neurons.append(
                    Neurona(id=Constructor.key_by_coord(x, y), umbral=0.0)
                )

        input_neurons: list[Neurona] = []
        for idx in range(n_input):
            input_neurons.append(NeuronaEntrada(id=f"inp_{idx}"))

        all_neurons: list[Neurona] = tissue_neurons + input_neurons
        self.brain = Brain(neuronas=all_neurons)

        # --- Regions ---
        region_tissue = Region(nombre="tissue")
        for n in tissue_neurons:
            region_tissue.agregar(n)

        region_input = Region(nombre="input")
        for n in input_neurons:
            region_input.agregar(n)

        self.regiones = {
            "tissue": region_tissue,
            "input": region_input,
        }

        # --- Apply wiring mask to tissue ---
        constructor = Constructor()
        constructor.aplicar_mascara_2d(
            self.brain,
            self.width,
            self.height,
            mask,
            random_weights=self._random_weights,
        )

        # --- Add input dendrite to each tissue neuron ---
        for tissue_n in tissue_neurons:
            sinapsis_list: list[Sinapsis] = []
            for inp_n in input_neurons:
                peso = random.uniform(0.2, 1.0)
                sinapsis_list.append(
                    Sinapsis(neurona_entrante=inp_n, peso=peso)
                )
            dendrita = Dendrita(
                sinapsis=sinapsis_list, peso=self.input_dendrite_weight
            )
            tissue_n.dendritas.append(dendrita)

        # --- Initialize tissue with random values ---
        for n in tissue_neurons:
            n.activar_external(random.random())

        # --- Compile ---
        self.brain_tensor = ConstructorTensor.compilar(
            self.brain,
            max_active_steps=self.max_active_steps,
            refractory_steps=self.refractory_steps,
            adaptation_enabled=self.adaptation_enabled,
        )

        # --- Pre-render characters ---
        self._char_images = {}
        for char in set(self.input_text):
            self._char_images[char] = render_char(
                char, self.input_resolution,
                font_id=self._font_id, font_size=self._font_size,
            )

        # --- Frame tracking ---
        self._char_index = 0
        self._frame_in_char = 0
        self._in_gap = False
        self._rng = np.random.default_rng()

        # --- Project initial frame ---
        self._generate_and_project()

    def _generate_and_project(self) -> None:
        """Generate the current input frame with noise and project onto input neurons."""
        res = self.input_resolution

        if not self.input_text:
            frame = self._rng.integers(0, 2, size=(res, res)).astype(np.float64)
        elif self._in_gap:
            if self.inter_char_noise:
                frame = self._rng.integers(0, 2, size=(res, res)).astype(np.float64)
            else:
                frame = np.zeros((res, res), dtype=np.float64)
        else:
            char = self.input_text[self._char_index]
            base = self._char_images[char]
            frame = base.copy()

            if self.shift_noise_enabled:
                frame = apply_shift_noise(frame, self._rng)
            if self.white_noise_enabled:
                frame = apply_white_noise(frame, noise_prob=self.noise_prob, rng=self._rng)

        self._current_input_frame = frame

        if self.brain_tensor is not None:
            flat = torch.from_numpy(frame.flatten()).float()
            start = self._input_start_idx
            end = start + len(flat)
            self.brain_tensor.valores[start:end] = flat

    def step(self) -> dict[str, Any]:
        """One step: generate frame, project, process, learn, advance counter."""
        self._generate_and_project()
        self.brain_tensor.procesar()

        if self.learning_enabled and self.brain_tensor is not None:
            self.brain_tensor.learn(lr=self.learning_rate)

        self.generation += 1

        if self.input_text:
            self._frame_in_char += 1
            if self._in_gap:
                if self._frame_in_char >= self.frames_per_char:
                    self._in_gap = False
                    self._frame_in_char = 0
                    self._char_index = (self._char_index + 1) % len(self.input_text)
            elif self._frame_in_char >= self.frames_per_char:
                self._in_gap = True
                self._frame_in_char = 0

        return {
            "type": "frame",
            "generation": self.generation,
            "grid": self.get_frame(),
            "stats": self.get_stats(),
        }

    def step_n(self, count: int) -> dict[str, Any]:
        """N steps — must loop because input changes each step."""
        result: dict[str, Any] = {}
        for _ in range(count):
            result = self.step()
        return result

    def click(self, x: int, y: int) -> None:
        """Toggle a tissue neuron."""
        idx = y * self.width + x
        if self.brain_tensor and 0 <= idx < self._input_start_idx:
            current = self.brain_tensor.valores[idx].item()
            self.brain_tensor.set_valor(idx, 0.0 if current >= 0.5 else 1.0)

    def get_frame(self) -> list[list[float]]:
        """Return the tissue grid (excludes input neurons)."""
        if self.brain_tensor:
            return self.brain_tensor.get_grid(self.width, self.height)
        return super().get_frame()

    def get_tension_frame(self) -> list[list[float]] | None:
        """Return the tissue tension grid."""
        if self.brain_tensor:
            return self.brain_tensor.get_tension_grid(self.width, self.height)
        return None

    def get_input_frame(self) -> list[list[float]] | None:
        """Return the current input image being projected."""
        if self._current_input_frame is not None:
            return self._current_input_frame.tolist()
        return None

    def get_stats(self) -> dict[str, Any]:
        """Stats including current character and frame info."""
        if self.brain_tensor is None:
            return super().get_stats()

        vals = self.brain_tensor.valores[: self._input_start_idx]
        active = int((vals > 0.5).sum().item())

        if not self.input_text:
            current_char = ""
        elif self._in_gap:
            current_char = "gap"
        else:
            current_char = self.input_text[self._char_index]

        return {
            "active_cells": active,
            "steps": self.generation,
            "current_char": current_char,
            "char_index": self._char_index,
            "frame_in_char": self._frame_in_char,
            "frames_per_char": self.frames_per_char,
            "input_resolution": self.input_resolution,
        }

    def inspect(self, x: int, y: int) -> dict[str, Any]:
        """Return connection weights for a tissue neuron.

        Reads ALL weights from brain_tensor (the live, trained copy)
        rather than the original Brain object.  This is critical because
        learning updates brain_tensor.pesos_sinapsis only — the original
        Python Sinapsis objects are never touched.
        """
        if self.brain_tensor is None:
            result = super().inspect(x, y)
            result["input_weight_grid"] = None
            return result

        neuron_idx = y * self.width + x
        n_input = self.input_resolution * self.input_resolution
        input_start = self._input_start_idx
        input_end = input_start + n_input

        sources = self.brain_tensor.indices_fuente[neuron_idx]
        weights = self.brain_tensor.pesos_sinapsis[neuron_idx]
        dend_weights = self.brain_tensor.pesos_dendrita[neuron_idx]
        valid = self.brain_tensor.mascara_valida[neuron_idx]
        dend_ids = self.brain_tensor.dendrita_ids[neuron_idx]

        total_sinapsis = int(valid.sum().item())
        total_dendritas = int(dend_ids[valid].unique().numel()) if total_sinapsis > 0 else 0

        # Accumulate effective weights for tissue (daemon) connections
        tissue_pesos: dict[int, float] = {}
        input_weights: list[float] = [0.0] * n_input

        for i in range(sources.shape[0]):
            if not valid[i]:
                continue
            src = sources[i].item()
            w = weights[i].item()
            dw = dend_weights[i].item()

            if input_start <= src < input_end:
                input_weights[src - input_start] = w
            elif src < input_start:
                effective = w * dw
                tissue_pesos[src] = tissue_pesos.get(src, 0.0) + effective

        for src in tissue_pesos:
            tissue_pesos[src] = max(-1.0, min(1.0, tissue_pesos[src]))

        # Build weight_grid (daemon connections)
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

        # Build input_weight_grid
        res = self.input_resolution
        input_grid: list[list[float]] = []
        for r in range(res):
            input_grid.append(input_weights[r * res : (r + 1) * res])

        return {
            "type": "connections",
            "x": x,
            "y": y,
            "total_dendritas": total_dendritas,
            "total_sinapsis": total_sinapsis,
            "weight_grid": weight_grid,
            "input_weight_grid": input_grid,
            "input_weight_width": res,
            "input_weight_height": res,
        }

    def update_config(self, config: dict[str, Any]) -> bool:
        """Update parameters on a running experiment.

        Soft params (learning, noise, text, font) are applied in-place,
        preserving both neuron activations and learned synapse weights.

        Hard params (dimensions, mask, resolution, dendrite/daemon weights)
        trigger a full reconnect that preserves activations but rebuilds
        connectivity (learned weights are lost).

        Returns True if only soft updates were needed.
        """
        if self.brain_tensor is None:
            return False

        hard_keys = {
            "width", "height", "mask", "input_resolution",
            "input_dendrite_weight", "deamon_exc_weight", "deamon_inh_weight",
        }
        needs_reconnect = any(
            k in config and config[k] != self._config.get(k)
            for k in hard_keys
        )

        if needs_reconnect:
            self.reconnect(config)
            return False

        if "learning" in config:
            self.learning_enabled = config["learning"]
        if "learning_rate" in config:
            self.learning_rate = config["learning_rate"]
        if "spike_adaptation" in config:
            self.adaptation_enabled = config["spike_adaptation"]
            if self.brain_tensor is not None:
                self.brain_tensor.adaptation_enabled = self.adaptation_enabled
        if "max_active_steps" in config:
            self.max_active_steps = config["max_active_steps"]
            if self.brain_tensor is not None:
                self.brain_tensor.max_active_steps = self.max_active_steps
        if "refractory_steps" in config:
            self.refractory_steps = config["refractory_steps"]
            if self.brain_tensor is not None:
                self.brain_tensor.refractory_steps = self.refractory_steps
        if "white_noise" in config:
            self.white_noise_enabled = config["white_noise"]
        if "noise_prob" in config:
            self.noise_prob = config["noise_prob"]
        if "shift_noise" in config:
            self.shift_noise_enabled = config["shift_noise"]
        if "inter_char_noise" in config:
            self.inter_char_noise = config["inter_char_noise"]
        if "frames_per_char" in config:
            self.frames_per_char = max(1, config["frames_per_char"])

        font_changed = False
        if "font" in config and config["font"] != self._font_id:
            self._font_id = config["font"]
            font_changed = True
        if "font_size" in config and config["font_size"] != self._font_size:
            self._font_size = config["font_size"]
            font_changed = True

        text_changed = False
        if "input_text" in config:
            new_text = config["input_text"] or ""
            if new_text != self.input_text:
                self.input_text = new_text
                text_changed = True
                self._char_index = 0
                self._frame_in_char = 0
                self._in_gap = False

        if font_changed or text_changed:
            self._char_images = {}
            for char in set(self.input_text):
                self._char_images[char] = render_char(
                    char, self.input_resolution,
                    font_id=self._font_id, font_size=self._font_size,
                )

        self._config.update(config)
        return True

    def reconnect(self, config: dict[str, Any]) -> None:
        """Rebuild with new mask/settings, preserving tissue neuron values."""
        if self.brain_tensor is None:
            return

        saved_values = self.brain_tensor.valores[: self._input_start_idx].clone()

        saved_char_index = self._char_index
        saved_frame_in_char = self._frame_in_char
        saved_in_gap = self._in_gap
        saved_generation = self.generation

        merged = {**self._config, **config}
        self.setup(merged)

        n_restore = min(len(saved_values), self._input_start_idx)
        self.brain_tensor.valores[:n_restore] = saved_values[:n_restore]

        if self.input_text:
            self._char_index = saved_char_index % len(self.input_text)
        else:
            self._char_index = 0
        self._frame_in_char = saved_frame_in_char
        self._in_gap = saved_in_gap
        self.generation = saved_generation

    def reset(self) -> None:
        """Reset the experiment with the same configuration."""
        self.setup(self._config)

    def is_complete(self) -> bool:
        """Dynamic SOM never ends."""
        return False

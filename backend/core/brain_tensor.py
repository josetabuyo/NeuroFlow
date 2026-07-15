"""BrainTensor — Neural network as tensors, vectorized processing.

Parallel equivalent of Brain. All operations are PyTorch tensor operations:
no Python loops over neurons/dendrites/synapses.

Main tensors:
  V  [N]          — current values of each neuron (+1 zero neuron if border)
  W  [N, max_syn] — synaptic weights
  C  [N, max_syn] — source neuron indices (connectivity)
  Dp [N, max_syn] — dendrite weight per synapse
  M  [N, max_syn] — valid synapse mask (bool)
  Di [N, max_syn] — dendrite ID per synapse (for segment_mean)
  U  [N]          — activation thresholds
  Em [N]          — NeuronaEntrada mask (do not process)
"""

from __future__ import annotations

import torch


class BrainTensor:
    """Neural network as tensors — vectorized processing."""

    def __init__(
        self,
        valores: torch.Tensor,
        pesos_sinapsis: torch.Tensor,
        indices_fuente: torch.LongTensor,
        pesos_dendrita: torch.Tensor,
        mascara_valida: torch.BoolTensor,
        dendrita_ids: torch.LongTensor,
        max_dendritas: int,
        umbrales: torch.Tensor,
        mascara_entrada: torch.BoolTensor,
        n_real: int,
        device: str = "cpu",
        max_active_steps: int = 5,
        refractory_steps: int = 5,
        adaptation_enabled: bool = False,
        process_mode: str = "min_vs_max",
        tension_fn: str = "",
        tension_fn_param: float = 1.0,
        tension_fns: list[tuple[str, float]] | None = None,
        region_specs: list[tuple[int, int, str, list]] | None = None,
        es_cross_region: torch.BoolTensor | None = None,
        dendrite_grupo_ids: torch.LongTensor | None = None,
    ) -> None:
        self.device = device
        self.process_mode = process_mode
        # Support both legacy single fn and composable list
        if tension_fns is not None:
            self.tension_fns = tension_fns
        elif tension_fn:
            self.tension_fns = [(tension_fn, tension_fn_param)]
        else:
            self.tension_fns = []
        # Per-region specs: list of (start, end, process_mode, tension_fns)
        # When set, overrides the global process_mode/tension_fns for each neuron range.
        self.region_specs: list[tuple[int, int, str, list]] = region_specs or []
        # n_real = number of actual neurons from the Brain
        # N = total including possible border zero neuron
        self.n_real = n_real
        self.N = valores.shape[0]

        self.valores = valores.to(device)
        self.pesos_sinapsis = pesos_sinapsis.to(device)
        self.indices_fuente = indices_fuente.to(device)
        self.pesos_dendrita = pesos_dendrita.to(device)
        self.mascara_valida = mascara_valida.to(device)
        self.dendrita_ids = dendrita_ids.to(device)
        self.max_dendritas = max_dendritas
        self.umbrales = umbrales.to(device)
        self.mascara_entrada = mascara_entrada.to(device)

        # Spike frequency adaptation: ON/OFF cycle
        #   active_counts tracks consecutive active steps (ON phase)
        #   refractory_remaining counts down forced-off steps (OFF phase, >0 = in refractory)
        self.adaptation_enabled = adaptation_enabled
        self.max_active_steps = max_active_steps
        self.refractory_steps = refractory_steps
        self.active_counts = torch.zeros(self.N, dtype=torch.long, device=device)
        self.refractory_remaining = torch.zeros(self.N, dtype=torch.long, device=device)

        # Safe dendrite IDs: invalid synapses point to a trash column (max_dendritas)
        # so they don't corrupt valid dendrite data during scatter operations.
        self._safe_dend_ids = self.dendrita_ids.clone()
        self._safe_dend_ids[~self.mascara_valida] = self.max_dendritas

        # Pre-compute per-dendrite weights [N, max_dend] and dendrite mask
        self._dend_pesos, self._dendrita_mascara = self._precompute_dendrite_info()

        # Cross-region synapse mask: source neuron lives in a different region
        # than the destination (built from connection spans). group_avg uses it to
        # separate distant dendrites from local ones. Learning rate per synapse is
        # supplied externally via learn(lr_per_syn).
        NR = n_real
        max_syn = pesos_sinapsis.shape[1] if pesos_sinapsis.ndim > 1 else 1
        if es_cross_region is not None:
            self.es_cross_region = es_cross_region.to(device)
        else:
            self.es_cross_region = torch.zeros(NR, max_syn, dtype=torch.bool, device=device)

        # Per-dendrite mask: True if that dendrite is a distant (cross-region) connection
        self._is_input_dendrite = self._precompute_input_dendrite_mask()

        # Per-dendrite declared-group index [n_real, max_dend]: dendrites sharing a
        # group get averaged together before group_avg splits groups by sign (see
        # Dendrita.grupo_id / ConstructorTensor). Absent → every dendrite is its own
        # singleton group (identity default, matches "no declared grouping").
        if dendrite_grupo_ids is not None:
            self._dendrite_grupo_ids = dendrite_grupo_ids.to(device)
        else:
            self._dendrite_grupo_ids = torch.arange(
                self.max_dendritas, device=device
            ).unsqueeze(0).expand(n_real, -1)
        self.max_grupos = int(self._dendrite_grupo_ids.max().item()) + 1 if self.max_dendritas > 0 else 1

        # Tension values (updated each procesar() call)
        self.tensiones = torch.zeros(self.N, device=device)

        # Soft activation mask: neurons where activation = clamp(tension, 0, 1) instead of threshold
        self.soft_mask = self._build_soft_mask()

    def _build_soft_mask(self) -> torch.BoolTensor:
        """Per-neuron bool: True means activation = clamp(tension, 0, 1) instead of threshold."""
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        for spec in self.region_specs:
            if len(spec) >= 5 and spec[4]:
                mask[spec[0]:spec[1]] = True
        return mask

    def _precompute_input_dendrite_mask(self) -> torch.BoolTensor:
        """Per-dendrite boolean: True if the dendrite is a distant (cross-region) connection."""
        NR = self._safe_dend_ids.shape[0]
        expanded = self.max_dendritas + 1
        distant_marker = self.es_cross_region.float()
        is_inp = torch.zeros(NR, expanded, device=self.device)
        is_inp.scatter_add_(1, self._safe_dend_ids, distant_marker)
        return (is_inp[:, :self.max_dendritas] > 0)

    def _precompute_dendrite_info(self) -> tuple[torch.Tensor, torch.BoolTensor]:
        """Pre-compute dendrite weights and validity mask.

        Uses a trash column (index max_dendritas) to safely scatter invalid synapses
        without corrupting valid dendrite data.
        """
        N = self._safe_dend_ids.shape[0]
        expanded = self.max_dendritas + 1  # +1 for trash column

        # Dendrite weights: scatter weights from valid synapses
        dend_pesos = torch.zeros(N, expanded, device=self.device)
        dend_pesos.scatter_(1, self._safe_dend_ids, self.pesos_dendrita)
        dend_pesos = dend_pesos[:, :self.max_dendritas]

        # Dendrite mask: a dendrite is valid if it has at least one valid synapse
        conteos = torch.zeros(N, expanded, device=self.device)
        conteos.scatter_add_(1, self._safe_dend_ids, self.mascara_valida.float())
        dendrita_mascara = conteos[:, :self.max_dendritas] > 0

        return dend_pesos, dendrita_mascara

    def _apply_tension_fns(self, tension: torch.Tensor, fns: list) -> torch.Tensor:
        result = torch.zeros_like(tension)
        for fn_name, coeff in fns:
            if fn_name == "x":
                result = result + coeff * tension
            elif fn_name == "b":
                result = result + coeff
            elif fn_name.startswith("x_pow_"):
                exp = int(fn_name.split("_pow_")[1])
                result = result + coeff * tension.pow(exp)
        return result.clamp(-1.0, 1.0)

    def _compute_tension(self, dpc: torch.Tensor, mode: str, r_start: int, r_end: int) -> torch.Tensor:
        """Compute tension for a slice of neurons [r_start:r_end] using dpc [slice_len, max_dend]."""
        if mode == "sum":
            return dpc.sum(dim=1).clamp(-1.0, 1.0)

        if mode in ("avg_vs_avg", "avg_vs_avg_normalized"):
            pos_mask = dpc > 0
            neg_mask = dpc < 0
            pos_avg = (dpc * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1.0)
            neg_avg = (dpc * neg_mask).sum(dim=1) / neg_mask.sum(dim=1).clamp(min=1.0)
            raw = pos_avg + neg_avg
            if mode == "avg_vs_avg_normalized":
                normalizer = (pos_avg - neg_avg).clamp(min=1e-8)
                return (raw / normalizer).clamp(-1.0, 1.0)
            return raw.clamp(-1.0, 1.0)

        if mode == "pos_vs_neg":
            max_pos = dpc.clamp(min=0.0).max(dim=1).values
            neg_mask = dpc < 0
            neg_avg = (dpc * neg_mask).sum(dim=1) / neg_mask.sum(dim=1).clamp(min=1.0)
            return (max_pos + neg_avg).clamp(-1.0, 1.0)

        if mode == "group_avg":
            valid = self._dendrita_mascara[r_start:r_end]
            pesos = self._dend_pesos[r_start:r_end]
            grupo_ids = self._dendrite_grupo_ids[r_start:r_end]
            rows = grupo_ids.shape[0]
            expanded = self.max_grupos + 1  # +1 trash column, mirrors _precompute_dendrite_info

            safe_grupo_ids = grupo_ids.clone()
            safe_grupo_ids[~valid] = self.max_grupos

            # Level 1: average dendrites sharing a declared grupo_id (one
            # deamon groups[] entry, or one nerve/full connection) into a
            # single per-group value.
            valid_f = valid.float()
            group_sum = torch.zeros(rows, expanded, device=self.device)
            group_sum.scatter_add_(1, safe_grupo_ids, dpc)
            group_count = torch.zeros(rows, expanded, device=self.device)
            group_count.scatter_add_(1, safe_grupo_ids, valid_f)
            group_avg = (group_sum / group_count.clamp(min=1.0))[:, :self.max_grupos]
            group_present = group_count[:, :self.max_grupos] > 0

            # Each group's polarity is intrinsic to its declared weight (not the
            # runtime activation value) — scatter the raw dendrite weight to
            # recover one representative sign per group.
            peso_sum = torch.zeros(rows, expanded, device=self.device)
            peso_sum.scatter_add_(1, safe_grupo_ids, pesos * valid_f)
            group_peso = (peso_sum / group_count.clamp(min=1.0))[:, :self.max_grupos]

            # Level 2: average the present groups' values by sign — one vote
            # per group, not per raw dendrite — then diff (neg side is
            # already negative-valued).
            def _polarity_avg(sign_mask: torch.BoolTensor) -> torch.Tensor:
                m = sign_mask & group_present
                count = m.sum(dim=1).clamp(min=1.0)
                return (group_avg * m).sum(dim=1) / count

            pos_avg = _polarity_avg(group_peso > 0)
            neg_avg = _polarity_avg(group_peso < 0)
            return (pos_avg + neg_avg).clamp(-1.0, 1.0)

        # min_vs_max (default)
        max_vals = dpc.max(dim=1).values.clamp(min=0.0)
        min_vals = dpc.min(dim=1).values.clamp(max=0.0)
        return (max_vals + min_vals).clamp(-1.0, 1.0)

    def procesar(self) -> None:
        """A full vectorized step.

        1. Gather: obtain source neuron values
        2. Synapse: 1 - |weight - input|
        3. Average per dendrite (segment mean with scatter_add_)
        4. Multiply by dendrite weight
        5. Combine dendrites (process_mode: min_vs_max or sum) → tension
        6. Activate: tension > threshold
        7. Preserve NeuronaEntrada (do not touch their values)
        """
        NR = self.n_real  # real neurons (synapse tensors have NR rows)
        expanded = self.max_dendritas + 1

        # 1. Gather: read source neuron values (indices may point to zero neuron at N)
        entradas = self.valores[self.indices_fuente]  # [NR, max_syn]

        # 2. Synapse processing: 1 - |weight - input|, masked
        syn_valores = (1.0 - torch.abs(self.pesos_sinapsis - entradas)) * self.mascara_valida

        # 3. Segment mean: average synapse values per dendrite
        # Use safe IDs so invalid synapses scatter to trash column
        sumas = torch.zeros(NR, expanded, device=self.device)
        conteos = torch.zeros(NR, expanded, device=self.device)

        sumas.scatter_add_(1, self._safe_dend_ids, syn_valores)
        conteos.scatter_add_(1, self._safe_dend_ids, self.mascara_valida.float())

        # Discard trash column
        sumas = sumas[:, :self.max_dendritas]
        conteos = conteos[:, :self.max_dendritas]

        promedios = sumas / conteos.clamp(min=1.0)  # [NR, max_dend]

        # 4. Multiply by dendrite weight
        dendrita_valores = promedios * self._dend_pesos  # [NR, max_dend]

        # 5. Combine dendrites (mode-dependent).
        # Invalid dendrites → 0 (neutral for both modes).
        dendrita_para_calc = dendrita_valores.where(self._dendrita_mascara, torch.zeros(1, device=self.device))

        tension = torch.zeros(NR, device=self.device)

        if self.region_specs:
            for r_start, r_end, r_mode, r_fns, *_ in self.region_specs:
                t = self._compute_tension(dendrita_para_calc[r_start:r_end], r_mode, r_start, r_end)
                if r_fns:
                    t = self._apply_tension_fns(t, r_fns)
                tension[r_start:r_end] = t
        else:
            t = self._compute_tension(dendrita_para_calc, self.process_mode, 0, NR)
            if self.tension_fns:
                t = self._apply_tension_fns(t, self.tension_fns)
            tension = t

        self.tensiones[:NR] = tension

        # 6. Activate: tension > threshold (only real neurons)
        umbrales_real = self.umbrales[:NR]
        mascara_real = self.mascara_entrada[:NR]
        valores_real = self.valores[:NR]

        # 6. Activate: binary threshold for normal neurons, tension for soft neurons
        binary = (tension > umbrales_real).float()
        soft = tension.clamp(0.0, 1.0)
        nuevos_valores = torch.where(self.soft_mask[:NR], soft, binary)

        # 7. Preserve NeuronaEntrada values
        self.valores[:NR] = torch.where(mascara_real, valores_real, nuevos_valores)

        # 8. Spike frequency adaptation: ON/OFF cycle
        if self.adaptation_enabled and self.max_active_steps > 0:
            procesables = ~mascara_real
            refr = self.refractory_remaining[:NR]
            ac = self.active_counts[:NR]
            zero_l = torch.zeros(1, dtype=torch.long, device=self.device)
            zero_f = torch.zeros(1, device=self.device)

            # Neurons in refractory period: force off, decrement counter
            in_refractory = procesables & (refr > 0)
            self.valores[:NR] = torch.where(in_refractory, zero_f, self.valores[:NR])
            self.refractory_remaining[:NR] = torch.where(in_refractory, refr - 1, refr)

            # For non-refractory processable neurons: track active streaks
            not_refr = procesables & (refr <= 0)
            activas = not_refr & (self.valores[:NR] > 0.5)
            inactivas = not_refr & (self.valores[:NR] <= 0.5)

            self.active_counts[:NR] = torch.where(activas, ac + 1, torch.where(inactivas, zero_l, ac))

            # Neurons that hit the limit: enter refractory period
            hit_limit = not_refr & (self.active_counts[:NR] >= self.max_active_steps)
            self.valores[:NR] = torch.where(hit_limit, zero_f, self.valores[:NR])
            self.active_counts[:NR] = torch.where(hit_limit, zero_l, self.active_counts[:NR])
            self.refractory_remaining[:NR] = torch.where(
                hit_limit,
                torch.full((1,), self.refractory_steps, dtype=torch.long, device=self.device),
                self.refractory_remaining[:NR],
            )
        

    def learn(
        self,
        lr_per_syn: torch.Tensor,
        excl_lo: torch.Tensor | None = None,
        excl_hi: torch.Tensor | None = None,
    ) -> None:
        """Tension-modulated Hebbian learning with a per-synapse learning rate.

        Rule: dW = lr_per_syn * tension * (source_value - weight)
          - lr_per_syn [NR, max_syn]: effective learning rate per synapse. It is
            built once in setup() (intra-region wiring rate vs. per-connection rate)
            and updated on soft-updates. A 0.0 entry freezes that synapse.
          - excl_lo/excl_hi [NR, max_syn]: per-synapse dead-zone bounds — skip the
            update where the current weight lies in [excl_lo, excl_hi] for that
            synapse. Any learning source (intra-region, nerve, or full connection)
            can configure its own range; synapses without one get a sentinel
            (lo > hi) that never matches.
        """
        NR = self.n_real

        source_vals = self.valores[self.indices_fuente]  # [NR, max_syn]
        tension = self.tensiones[:NR].unsqueeze(1)       # [NR, 1]

        delta = lr_per_syn * tension * (source_vals - self.pesos_sinapsis)

        if excl_lo is not None and excl_hi is not None:
            in_range = (self.pesos_sinapsis >= excl_lo) & (self.pesos_sinapsis <= excl_hi)
            delta = delta.masked_fill(in_range, 0.0)

        self.pesos_sinapsis = (self.pesos_sinapsis + delta * self.mascara_valida).clamp(0.0, 1.0)

    def procesar_n(self, n: int) -> None:
        """N steps seguidos sin salir al Python loop."""
        for _ in range(n):
            self.procesar()

    def get_grid(self, width: int, height: int) -> list[list[float]]:
        """Convierte tensor de valores a grilla 2D."""
        return self.valores[:width * height].reshape(height, width).tolist()

    def get_tension_grid(self, width: int, height: int) -> list[list[float]]:
        """Convierte tensor de tensiones a grilla 2D."""
        return self.tensiones[:width * height].reshape(height, width).tolist()

    def get_valores(self) -> torch.Tensor:
        """Retorna el tensor de valores."""
        return self.valores

    def set_valor(self, idx: int, valor: float) -> None:
        """Modifica el valor de una neurona (para click/paint)."""
        self.valores[idx] = valor

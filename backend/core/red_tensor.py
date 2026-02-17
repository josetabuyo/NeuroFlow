"""RedTensor — Red neuronal como tensores, procesamiento vectorizado.

Equivalente paralelo de Red. Todas las operaciones son tensoriales PyTorch:
sin loops Python sobre neuronas/dendritas/sinapsis.

Tensores principales:
  V  [N]          — valores actuales de cada neurona (+1 zero neuron if border)
  W  [N, max_syn] — pesos sinápticos
  C  [N, max_syn] — índices de neurona fuente (conectividad)
  Dp [N, max_syn] — peso de dendrita para cada sinapsis
  M  [N, max_syn] — máscara de sinapsis válidas (bool)
  Di [N, max_syn] — ID de dendrita por sinapsis (para segment_mean)
  U  [N]          — umbrales de activación
  Em [N]          — máscara de NeuronaEntrada (no procesar)
"""

from __future__ import annotations

import torch


class RedTensor:
    """Red neuronal como tensores — procesamiento vectorizado."""

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
    ) -> None:
        self.device = device
        # n_real = number of actual neurons from the Red
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

        # Safe dendrite IDs: invalid synapses point to a trash column (max_dendritas)
        # so they don't corrupt valid dendrite data during scatter operations.
        self._safe_dend_ids = self.dendrita_ids.clone()
        self._safe_dend_ids[~self.mascara_valida] = self.max_dendritas

        # Pre-compute per-dendrite weights [N, max_dend] and dendrite mask
        self._dend_pesos, self._dendrita_mascara = self._precompute_dendrite_info()

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

    def procesar(self) -> None:
        """Un step completo vectorizado.

        1. Gather: obtener valores de neuronas fuente
        2. Sinapsis: 1 - |peso - entrada|
        3. Promedio por dendrita (segment mean con scatter_add_)
        4. Multiplicar por peso de dendrita
        5. Fuzzy OR: max(0, dendritas) + min(0, dendritas) → tensión
        6. Activar: tensión > umbral
        7. Preservar NeuronaEntrada (no tocar sus valores)
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

        # 5. Fuzzy OR: max(0, dendritas) + min(0, dendritas)
        # Legacy code initializes max_valor=0.0, min_valor=0.0 before iterating
        # dendrites, so max is at least 0 and min is at most 0.
        # For invalid dendrites, set values to 0 (neutral for both max and min).
        dendrita_para_calc = dendrita_valores.where(self._dendrita_mascara, torch.zeros(1, device=self.device))

        max_vals = dendrita_para_calc.max(dim=1).values.clamp(min=0.0)  # [NR]
        min_vals = dendrita_para_calc.min(dim=1).values.clamp(max=0.0)  # [NR]

        tension = (max_vals + min_vals).clamp(-1.0, 1.0)  # [NR]

        # 6. Activate: tension > threshold (only real neurons)
        umbrales_real = self.umbrales[:NR]
        mascara_real = self.mascara_entrada[:NR]
        valores_real = self.valores[:NR]

        nuevos_valores = (tension > umbrales_real).float()  # [NR]

        # 7. Preserve NeuronaEntrada values
        self.valores[:NR] = torch.where(mascara_real, valores_real, nuevos_valores)

    def procesar_n(self, n: int) -> None:
        """N steps seguidos sin salir al Python loop."""
        for _ in range(n):
            self.procesar()

    def get_grid(self, width: int, height: int) -> list[list[float]]:
        """Convierte tensor de valores a grilla 2D."""
        return self.valores[:width * height].reshape(height, width).tolist()

    def get_valores(self) -> torch.Tensor:
        """Retorna el tensor de valores."""
        return self.valores

    def set_valor(self, idx: int, valor: float) -> None:
        """Modifica el valor de una neurona (para click/paint)."""
        self.valores[idx] = valor

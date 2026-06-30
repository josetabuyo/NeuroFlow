"""ConstructorTensor — compiles a sequential Brain into a parallel BrainTensor.

Traverses the Brain ONCE and builds all the PyTorch tensors
needed for vectorized processing.

This is a setup step (O(N*S)), not a processing step.
It only runs once when starting the experiment.
"""

from __future__ import annotations

import numpy as np
import torch

from .brain import Brain
from .neurona import Neurona, NeuronaEntrada
from .brain_tensor import BrainTensor


class ConstructorTensor:
    """Compiles a sequential Brain into a parallel BrainTensor."""

    @staticmethod
    def compilar(
        brain: Brain,
        device: str = "cpu",
        max_active_steps: int = 5,
        refractory_steps: int = 5,
        adaptation_enabled: bool = False,
        process_mode: str = "min_vs_max",
        tension_fn: str = "",
        tension_fn_param: float = 1.0,
        tension_fns: list[tuple[str, float]] | None = None,
        connections: list[tuple[int, int, int, int]] | None = None,
        region_specs: list[tuple[int, int, str, list, bool]] | None = None,
    ) -> BrainTensor:
        """Convert a sequential Brain into a parallel BrainTensor.

        Args:
            brain: The sequential Brain with all neurons/dendrites/synapses configured.
            device: PyTorch device ("cpu" or "cuda").
            connections: cross-region spans as (src_start, src_end, dst_start, dst_end).
                A synapse is cross-region when its destination is in [dst_start, dst_end)
                and its source is in [src_start, src_end). Used to mark es_cross_region
                generically (group_avg needs it to separate distant from local dendrites).
            region_specs: per-region (start, end, process_mode, tension_fns, soft_activation) overrides.

        Returns:
            A BrainTensor ready for vectorized processing.
        """
        N = len(brain.neuronas)

        id_to_idx: dict[str, int] = {}
        for i, neurona in enumerate(brain.neuronas):
            id_to_idx[neurona.id] = i

        max_syn = 0
        max_dend = 0
        for neurona in brain.neuronas:
            total_syn = sum(len(d.sinapsis) for d in neurona.dendritas)
            max_syn = max(max_syn, total_syn)
            max_dend = max(max_dend, len(neurona.dendritas))

        if max_syn == 0:
            max_syn = 1
        if max_dend == 0:
            max_dend = 1

        valores_np = np.zeros(N, dtype=np.float32)
        umbrales_np = np.zeros(N, dtype=np.float32)
        entrada_np = np.zeros(N, dtype=np.bool_)
        pesos_s_np = np.zeros((N, max_syn), dtype=np.float32)
        indices_f_np = np.full((N, max_syn), N, dtype=np.int64)
        pesos_d_np = np.zeros((N, max_syn), dtype=np.float32)
        mascara_v_np = np.zeros((N, max_syn), dtype=np.bool_)
        dend_ids_np = np.zeros((N, max_syn), dtype=np.int64)

        for i, neurona in enumerate(brain.neuronas):
            valores_np[i] = neurona.valor
            umbrales_np[i] = neurona.umbral

            if isinstance(neurona, NeuronaEntrada):
                entrada_np[i] = True

            ps: list[float] = []
            fi: list[int] = []
            pd: list[float] = []
            di: list[int] = []

            for d_idx, dendrita in enumerate(neurona.dendritas):
                dw = dendrita.peso
                for sinapsis in dendrita.sinapsis:
                    ps.append(sinapsis.peso)
                    fi.append(id_to_idx.get(sinapsis.neurona_entrante.id, N))
                    pd.append(dw)
                    di.append(d_idx)

            k = len(ps)
            if k:
                pesos_s_np[i, :k] = ps
                indices_f_np[i, :k] = fi
                pesos_d_np[i, :k] = pd
                mascara_v_np[i, :k] = True
                dend_ids_np[i, :k] = di

        valores = torch.from_numpy(valores_np)
        umbrales = torch.from_numpy(umbrales_np)
        mascara_entrada = torch.from_numpy(entrada_np)
        pesos_sinapsis = torch.from_numpy(pesos_s_np)
        indices_fuente = torch.from_numpy(indices_f_np)
        pesos_dendrita = torch.from_numpy(pesos_d_np)
        mascara_valida = torch.from_numpy(mascara_v_np)
        dendrita_ids = torch.from_numpy(dend_ids_np)

        has_border = (indices_fuente == N).any().item()
        if has_border:
            valores = torch.cat([valores, torch.zeros(1)])
            umbrales = torch.cat([umbrales, torch.zeros(1)])
            mascara_entrada = torch.cat([mascara_entrada, torch.tensor([True])])
        else:
            indices_fuente = indices_fuente.clamp(0, N - 1)

        NR = N
        src_safe = indices_fuente.clamp(0, mascara_entrada.shape[0] - 1)

        # ── Cross-region synapse mask ──
        # A synapse is cross-region when (dst in [dst_start, dst_end)) and
        # (src in [src_start, src_end)) for any declared connection span.
        es_cross_region = torch.zeros(NR, max_syn, dtype=torch.bool)
        dst_idx = torch.arange(NR).unsqueeze(1).expand(NR, max_syn)
        for (src_start, src_end, dst_start, dst_end) in (connections or []):
            dst_in = (dst_idx >= dst_start) & (dst_idx < dst_end)
            src_in = (src_safe >= src_start) & (src_safe < src_end)
            es_cross_region |= dst_in & src_in & mascara_valida

        return BrainTensor(
            valores=valores,
            pesos_sinapsis=pesos_sinapsis,
            indices_fuente=indices_fuente,
            pesos_dendrita=pesos_dendrita,
            mascara_valida=mascara_valida,
            dendrita_ids=dendrita_ids,
            max_dendritas=max_dend,
            umbrales=umbrales,
            mascara_entrada=mascara_entrada,
            n_real=N,
            device=device,
            max_active_steps=max_active_steps,
            refractory_steps=refractory_steps,
            adaptation_enabled=adaptation_enabled,
            process_mode=process_mode,
            tension_fn=tension_fn,
            tension_fn_param=tension_fn_param,
            tension_fns=tension_fns,
            es_cross_region=es_cross_region,
            region_specs=region_specs,
        )

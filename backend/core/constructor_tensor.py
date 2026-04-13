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
    def compilar(brain: Brain, device: str = "cpu", max_active_steps: int = 5, refractory_steps: int = 5, adaptation_enabled: bool = False, process_mode: str = "min_vs_max", tension_fn: str = "", tension_fn_param: float = 1.0, tension_fns: list[tuple[str, float]] | None = None) -> BrainTensor:
        """Convert a sequential Brain into a parallel BrainTensor.

        Traverses the Brain ONCE and builds the tensors:
        1. Extract values from all neurons → tensor V [N]
        2. Extract synaptic weights → tensor W [N, max_syn]
        3. Extract connectivity (source neuron indices) → tensor C [N, max_syn]
        4. Extract dendrite weight per synapse → tensor Dp [N, max_syn]
        5. Extract valid synapse mask → tensor M [N, max_syn]
        6. Extract synapse-to-dendrite membership → tensor Di [N, max_syn]
        7. Extract thresholds → tensor U [N]
        8. Extract NeuronaEntrada mask → tensor Em [N]

        Args:
            brain: The sequential Brain with all neurons/dendrites/synapses configured.
            device: PyTorch device ("cpu" or "cuda").

        Returns:
            A BrainTensor ready for vectorized processing.
        """
        N = len(brain.neuronas)

        # Build neuron ID → index mapping
        id_to_idx: dict[str, int] = {}
        for i, neurona in enumerate(brain.neuronas):
            id_to_idx[neurona.id] = i

        # First pass: count max synapses per neuron and max dendrites
        max_syn = 0
        max_dend = 0
        for neurona in brain.neuronas:
            total_syn = sum(len(d.sinapsis) for d in neurona.dendritas)
            max_syn = max(max_syn, total_syn)
            max_dend = max(max_dend, len(neurona.dendritas))

        # Handle edge case: no synapses at all
        if max_syn == 0:
            max_syn = 1
        if max_dend == 0:
            max_dend = 1

        # Allocate numpy arrays for bulk fill (list appends + numpy slice
        # assignment are ~20x faster than individual PyTorch element writes)
        valores_np = np.zeros(N, dtype=np.float32)
        umbrales_np = np.zeros(N, dtype=np.float32)
        entrada_np = np.zeros(N, dtype=np.bool_)
        pesos_s_np = np.zeros((N, max_syn), dtype=np.float32)
        indices_f_np = np.full((N, max_syn), N, dtype=np.int64)
        pesos_d_np = np.zeros((N, max_syn), dtype=np.float32)
        mascara_v_np = np.zeros((N, max_syn), dtype=np.bool_)
        dend_ids_np = np.zeros((N, max_syn), dtype=np.int64)

        # Second pass: fill via Python lists, then bulk-assign per row
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

        # Convert to tensors in one shot (zero-copy from numpy)
        valores = torch.from_numpy(valores_np)
        umbrales = torch.from_numpy(umbrales_np)
        mascara_entrada = torch.from_numpy(entrada_np)
        pesos_sinapsis = torch.from_numpy(pesos_s_np)
        indices_fuente = torch.from_numpy(indices_f_np)
        pesos_dendrita = torch.from_numpy(pesos_d_np)
        mascara_valida = torch.from_numpy(mascara_v_np)
        dendrita_ids = torch.from_numpy(dend_ids_np)

        # Check if we need a zero neuron for border synapses
        has_border = (indices_fuente == N).any().item()
        if has_border:
            # Append a zero neuron (marked as NeuronaEntrada so it stays frozen)
            valores = torch.cat([valores, torch.zeros(1)])
            umbrales = torch.cat([umbrales, torch.zeros(1)])
            mascara_entrada = torch.cat([mascara_entrada, torch.tensor([True])])
            # indices_fuente already points to N which is the new zero neuron
        else:
            # Clamp any stray indices (shouldn't happen, but safety)
            indices_fuente = indices_fuente.clamp(0, N - 1)

        # ── Per-synapse type masks ──
        # Used by learn() to apply different learning rates per dendrite type.
        # Border synapses are excluded because mascara_valida is False for them.
        NR = N  # real neuron count before possible border extension
        src_safe = indices_fuente.clamp(0, mascara_entrada.shape[0] - 1)
        src_is_input = mascara_entrada[src_safe]  # [NR, max_syn]

        es_input_syn = src_is_input & mascara_valida           # source is NeuronaEntrada
        es_exc_syn   = (~src_is_input) & (pesos_dendrita >= 0) & mascara_valida
        es_inh_syn   = (~src_is_input) & (pesos_dendrita <  0) & mascara_valida

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
            es_exc_syn=es_exc_syn,
            es_inh_syn=es_inh_syn,
            es_input_syn=es_input_syn,
        )

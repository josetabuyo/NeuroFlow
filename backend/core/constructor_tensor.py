"""ConstructorTensor — compiles a sequential Red into a parallel RedTensor.

Traverses the Red ONCE and builds all the PyTorch tensors
needed for vectorized processing.

This is a setup step (O(N*S)), not a processing step.
It only runs once when starting the experiment.
"""

from __future__ import annotations

import torch

from .red import Red
from .neurona import Neurona, NeuronaEntrada
from .red_tensor import RedTensor


class ConstructorTensor:
    """Compiles a sequential Red into a parallel RedTensor."""

    @staticmethod
    def compilar(red: Red, device: str = "cpu") -> RedTensor:
        """Convert a sequential Red into a parallel RedTensor.

        Traverses the Red ONCE and builds the tensors:
        1. Extract values from all neurons → tensor V [N]
        2. Extract synaptic weights → tensor W [N, max_syn]
        3. Extract connectivity (source neuron indices) → tensor C [N, max_syn]
        4. Extract dendrite weight per synapse → tensor Dp [N, max_syn]
        5. Extract valid synapse mask → tensor M [N, max_syn]
        6. Extract synapse-to-dendrite membership → tensor Di [N, max_syn]
        7. Extract thresholds → tensor U [N]
        8. Extract NeuronaEntrada mask → tensor Em [N]

        Args:
            red: The sequential Red with all neurons/dendrites/synapses configured.
            device: PyTorch device ("cpu" or "cuda").

        Returns:
            A RedTensor ready for vectorized processing.
        """
        N = len(red.neuronas)

        # Build neuron ID → index mapping
        id_to_idx: dict[str, int] = {}
        for i, neurona in enumerate(red.neuronas):
            id_to_idx[neurona.id] = i

        # First pass: count max synapses per neuron and max dendrites
        max_syn = 0
        max_dend = 0
        for neurona in red.neuronas:
            total_syn = sum(len(d.sinapsis) for d in neurona.dendritas)
            max_syn = max(max_syn, total_syn)
            max_dend = max(max_dend, len(neurona.dendritas))

        # Handle edge case: no synapses at all
        if max_syn == 0:
            max_syn = 1
        if max_dend == 0:
            max_dend = 1

        # Allocate tensors
        valores = torch.zeros(N, dtype=torch.float32)
        pesos_sinapsis = torch.zeros(N, max_syn, dtype=torch.float32)
        indices_fuente = torch.zeros(N, max_syn, dtype=torch.long)
        pesos_dendrita = torch.zeros(N, max_syn, dtype=torch.float32)
        mascara_valida = torch.zeros(N, max_syn, dtype=torch.bool)
        dendrita_ids = torch.zeros(N, max_syn, dtype=torch.long)
        umbrales = torch.zeros(N, dtype=torch.float32)
        mascara_entrada = torch.zeros(N, dtype=torch.bool)

        # Second pass: fill tensors
        for i, neurona in enumerate(red.neuronas):
            valores[i] = neurona.valor
            umbrales[i] = neurona.umbral

            if isinstance(neurona, NeuronaEntrada):
                mascara_entrada[i] = True

            syn_idx = 0
            for d_idx, dendrita in enumerate(neurona.dendritas):
                for sinapsis in dendrita.sinapsis:
                    pesos_sinapsis[i, syn_idx] = sinapsis.peso
                    pesos_dendrita[i, syn_idx] = dendrita.peso
                    mascara_valida[i, syn_idx] = True
                    dendrita_ids[i, syn_idx] = d_idx

                    src_id = sinapsis.neurona_entrante.id
                    if src_id in id_to_idx:
                        indices_fuente[i, syn_idx] = id_to_idx[src_id]
                    else:
                        indices_fuente[i, syn_idx] = N

                    syn_idx += 1

        # Check if we need a zero neuron for border synapses
        has_border = (indices_fuente == N).any().item()
        if has_border:
            # Append a zero neuron
            valores = torch.cat([valores, torch.zeros(1)])
            umbrales = torch.cat([umbrales, torch.zeros(1)])
            mascara_entrada = torch.cat([mascara_entrada, torch.tensor([True])])
            # indices_fuente already points to N which is the new zero neuron
        else:
            # Clamp any stray indices (shouldn't happen, but safety)
            indices_fuente = indices_fuente.clamp(0, N - 1)

        return RedTensor(
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
        )

"""ConstructorTensor — compila una Red secuencial en una RedTensor paralela.

Recorre la Red UNA sola vez y construye todos los tensores PyTorch
necesarios para el procesamiento vectorizado.

Este es un paso de setup (O(N*S)), no de procesamiento.
Solo corre una vez al iniciar el experimento.
"""

from __future__ import annotations

import torch

from .red import Red
from .neurona import Neurona, NeuronaEntrada
from .red_tensor import RedTensor


class ConstructorTensor:
    """Compila una Red secuencial en una RedTensor paralela."""

    @staticmethod
    def compilar(red: Red, device: str = "cpu") -> RedTensor:
        """Convierte una Red secuencial en una RedTensor paralela.

        Recorre la Red UNA sola vez y construye los tensores:
        1. Extraer valores de todas las neuronas → tensor V [N]
        2. Extraer pesos sinápticos → tensor W [N, max_syn]
        3. Extraer conectividad (índices de neuronas fuente) → tensor C [N, max_syn]
        4. Extraer pesos de dendrita por sinapsis → tensor Dp [N, max_syn]
        5. Extraer máscara de sinapsis válidas → tensor M [N, max_syn]
        6. Extraer pertenencia de sinapsis a dendrita → tensor Di [N, max_syn]
        7. Extraer umbrales → tensor U [N]
        8. Extraer máscara de NeuronaEntrada → tensor Em [N]

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

                    # Resolve source neuron index
                    src_id = sinapsis.neurona_entrante.id
                    if src_id in id_to_idx:
                        indices_fuente[i, syn_idx] = id_to_idx[src_id]
                    else:
                        # Border neuron (not in the Red): created by conectar_filas
                        # with value 0, contributes 1 - |peso - 0| = 1 - peso.
                        # Point to a virtual zero neuron at index N.
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

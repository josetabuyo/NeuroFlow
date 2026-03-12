"""Constructor — factory for building networks + regions.

Creates neurons, groups them into regions,
builds connectivity (dendrites, synapses).
The resulting Brain is compiled to BrainTensor for parallel processing.
"""

from __future__ import annotations

import random

from .sinapsis import Sinapsis
from .dendrita import Dendrita
from .neurona import Neurona, NeuronaEntrada
from .brain import Brain
from .region import Region


class Constructor:
    """Factory/Builder for neural networks and regions."""

    @staticmethod
    def key_by_coord(x: int, y: int) -> str:
        """Generate neuron ID from coordinates."""
        return f"x{x}y{y}"

    def crear_grilla(
        self,
        width: int,
        height: int,
        filas_entrada: list[int],
        filas_salida: list[int],
        umbral: float = 0.0,
    ) -> tuple[Brain, dict[str, Region]]:
        """Create a neuron grid with input, output, and internal regions.

        Args:
            width: Grid width.
            height: Grid height.
            filas_entrada: Row indices that are NeuronaEntrada.
            filas_salida: Row indices that are output (normal Neurona, label only).
            umbral: Activation threshold for internal and output neurons.

        Returns:
            Tuple (Brain, dict of Regions).
        """
        neuronas: list[Neurona] = []
        region_entrada = Region(nombre="entrada")
        region_salida = Region(nombre="salida")
        region_interna = Region(nombre="interna")

        for y in range(height):
            for x in range(width):
                id_neurona = self.key_by_coord(x, y)

                if y in filas_entrada:
                    neurona: Neurona = NeuronaEntrada(id=id_neurona)
                    region_entrada.agregar(neurona)
                elif y in filas_salida:
                    neurona = Neurona(id=id_neurona, umbral=umbral)
                    region_salida.agregar(neurona)
                else:
                    neurona = Neurona(id=id_neurona, umbral=umbral)
                    region_interna.agregar(neurona)

                neuronas.append(neurona)

        brain = Brain(neuronas=neuronas)
        regiones = {
            "entrada": region_entrada,
            "salida": region_salida,
            "interna": region_interna,
        }
        return brain, regiones

    def conectar_filas(
        self,
        brain: Brain,
        fila_destino: int,
        width: int,
        height: int,
        mascara_relativa: list[tuple[int, int]],
        regla_pesos: list[list[float]],
        peso_dendrita: float = 1.0,
    ) -> None:
        """Connect neurons from a row with their neighbors according to a relative mask.

        Uses toroidal (wrap-around) topology: offsets that fall outside the
        grid wrap to the opposite edge via modular arithmetic.

        Args:
            brain: The network containing the neurons.
            fila_destino: Row index whose neurons will receive the connections.
            width: Grid width.
            height: Grid height.
            mascara_relativa: List of offsets (dx, dy) to find neighbors.
            regla_pesos: List of dendrites, each is a list of synapse weights.
                         Each weight list corresponds to a pattern to recognize.
            peso_dendrita: Weight for each created dendrite.
        """
        for x in range(width):
            neurona_destino = brain.get_neurona(self.key_by_coord(x, fila_destino))

            for pesos_sinapsis in regla_pesos:
                sinapsis_list: list[Sinapsis] = []

                for i, (dx, dy) in enumerate(mascara_relativa):
                    nx = (x + dx) % width
                    ny = (fila_destino + dy) % height
                    neurona_fuente = brain.get_neurona(self.key_by_coord(nx, ny))

                    peso_sinapsis = pesos_sinapsis[i] if i < len(pesos_sinapsis) else 0.0
                    sinapsis_list.append(
                        Sinapsis(neurona_entrante=neurona_fuente, peso=peso_sinapsis)
                    )

                dendrita = Dendrita(sinapsis=sinapsis_list, peso=peso_dendrita)
                neurona_destino.dendritas.append(dendrita)

    def aplicar_regla_wolfram(
        self,
        brain: Brain,
        regla: int,
        fila_destino: int,
        width: int,
        height: int,
    ) -> None:
        """Configure a row's dendrites according to a Wolfram rule.

        For each pattern (0 to 7) that produces output = 1 in the rule,
        creates a dendrite with 3 synapses whose weights encode the pattern.

        Args:
            brain: The network with the neurons.
            regla: Wolfram rule number (0-255).
            fila_destino: Row to configure.
            width: Grid width.
            height: Grid height.
        """
        patrones_activos: list[list[float]] = []
        for patron in range(8):
            if regla & (1 << patron):
                # Convert 3-bit pattern to synapse weights
                # Bit 2 = left, Bit 1 = center, Bit 0 = right
                izq = float((patron >> 2) & 1)
                cen = float((patron >> 1) & 1)
                der = float(patron & 1)
                patrones_activos.append([izq, cen, der])

        mascara = [(-1, 1), (0, 1), (1, 1)]  # left, center, right from row below
        self.conectar_filas(
            brain=brain,
            fila_destino=fila_destino,
            width=width,
            height=height,
            mascara_relativa=mascara,
            regla_pesos=patrones_activos,
            peso_dendrita=1.0,
        )

    def aplicar_mascara_2d(
        self,
        brain: Brain,
        width: int,
        height: int,
        mascara: list[dict[str, object]],
        *,
        random_weights: bool = True,
    ) -> None:
        """Apply a connection mask to every neuron in the 2D grid.

        Args:
            brain: The network with the neurons.
            width: Grid width.
            height: Grid height.
            mascara: List of dendrite definitions. Each one:
                {
                    "peso_dendrita": float,
                    "offsets": [(dx, dy), ...],
                    "pesos_sinapsis": [float, ...],  # optional, explicit weights
                }
            random_weights: If True, synapses without explicit pesos_sinapsis
                get random weights in [0.2, 1.0]. If False, they get 1.0.
                Dendrites with pesos_sinapsis always use those exact values.
        """
        for y in range(height):
            for x in range(width):
                neurona_destino = brain.get_neurona(self.key_by_coord(x, y))

                for def_dendrita in mascara:
                    peso_dendrita: float = def_dendrita["peso_dendrita"]  # type: ignore[assignment]
                    offsets: list[tuple[int, int]] = def_dendrita["offsets"]  # type: ignore[assignment]
                    pesos_explicitos: list[float] | None = def_dendrita.get("pesos_sinapsis")  # type: ignore[assignment]

                    sinapsis_list: list[Sinapsis] = []
                    for i, (dx, dy) in enumerate(offsets):
                        nx = (x + dx) % width
                        ny = (y + dy) % height
                        neurona_fuente = brain.get_neurona(
                            self.key_by_coord(nx, ny)
                        )

                        if pesos_explicitos is not None:
                            peso = pesos_explicitos[i]
                        elif random_weights:
                            peso = random.uniform(0.2, 1.0)
                        else:
                            peso = 1.0
                        sinapsis_list.append(
                            Sinapsis(neurona_entrante=neurona_fuente, peso=peso)
                        )

                    if sinapsis_list:
                        dendrita = Dendrita(
                            sinapsis=sinapsis_list, peso=peso_dendrita
                        )
                        neurona_destino.dendritas.append(dendrita)

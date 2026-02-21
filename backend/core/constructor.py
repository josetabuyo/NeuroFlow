"""Constructor — factory for building networks + regions.

Creates neurons, groups them into regions,
builds connectivity (dendrites, synapses).
The resulting Red is compiled to RedTensor for parallel processing.
"""

from __future__ import annotations

import random

from .sinapsis import Sinapsis
from .dendrita import Dendrita
from .neurona import Neurona, NeuronaEntrada
from .red import Red
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
    ) -> tuple[Red, dict[str, Region]]:
        """Create a neuron grid with input, output, and internal regions.

        Args:
            width: Grid width.
            height: Grid height.
            filas_entrada: Row indices that are NeuronaEntrada.
            filas_salida: Row indices that are output (normal Neurona, label only).
            umbral: Activation threshold for internal and output neurons.

        Returns:
            Tuple (Red, dict of Regions).
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

        red = Red(neuronas=neuronas)
        regiones = {
            "entrada": region_entrada,
            "salida": region_salida,
            "interna": region_interna,
        }
        return red, regiones

    def conectar_filas(
        self,
        red: Red,
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
            red: The network containing the neurons.
            fila_destino: Row index whose neurons will receive the connections.
            width: Grid width.
            height: Grid height.
            mascara_relativa: List of offsets (dx, dy) to find neighbors.
            regla_pesos: List of dendrites, each is a list of synapse weights.
                         Each weight list corresponds to a pattern to recognize.
            peso_dendrita: Weight for each created dendrite.
        """
        for x in range(width):
            neurona_destino = red.get_neurona(self.key_by_coord(x, fila_destino))

            for pesos_sinapsis in regla_pesos:
                sinapsis_list: list[Sinapsis] = []

                for i, (dx, dy) in enumerate(mascara_relativa):
                    nx = (x + dx) % width
                    ny = (fila_destino + dy) % height
                    neurona_fuente = red.get_neurona(self.key_by_coord(nx, ny))

                    peso_sinapsis = pesos_sinapsis[i] if i < len(pesos_sinapsis) else 0.0
                    sinapsis_list.append(
                        Sinapsis(neurona_entrante=neurona_fuente, peso=peso_sinapsis)
                    )

                dendrita = Dendrita(sinapsis=sinapsis_list, peso=peso_dendrita)
                neurona_destino.dendritas.append(dendrita)

    def aplicar_regla_wolfram(
        self,
        red: Red,
        regla: int,
        fila_destino: int,
        width: int,
        height: int,
    ) -> None:
        """Configure a row's dendrites according to a Wolfram rule.

        For each pattern (0 to 7) that produces output = 1 in the rule,
        creates a dendrite with 3 synapses whose weights encode the pattern.

        Args:
            red: The network with the neurons.
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
            red=red,
            fila_destino=fila_destino,
            width=width,
            height=height,
            mascara_relativa=mascara,
            regla_pesos=patrones_activos,
            peso_dendrita=1.0,
        )

    def balancear_pesos(
        self,
        neuronas: list[Neurona],
        target: float = 0.0,
    ) -> None:
        """Shift the Fuzzy OR operating point by scaling synaptic weights.

        The Fuzzy OR computes: tension = max(D_exc) + min(D_inh).
        With similar synaptic weights (~0.6) in exc and inh, the Fuzzy OR
        is naturally balanced (tension ≈ 0).

        This method shifts that balance by uniformly scaling the synapses
        on the opposite side:

          target =  0.0 → no change (natural Kohonen dynamics)
          target > 0   → scales inhibitory synapses by (1 - target)
                         → excitatory bias (more neurons activate)
          target < 0   → scales excitatory synapses by (1 + target)
                         → inhibitory bias (fewer neurons activate)
          target = +1  → inhibition removed (all ON)
          target = -1  → excitation removed (all OFF)

        Args:
            neuronas: List of neurons to balance. NeuronaEntrada and neurons
                      without dendrites are skipped.
            target: Balance offset. 0.0 = no change. Useful range: [-1, 1].
        """
        if target == 0.0:
            return

        for neurona in neuronas:
            if isinstance(neurona, NeuronaEntrada):
                continue
            if not neurona.dendritas:
                continue

            if target > 0:
                factor = max(0.01, 1.0 - target)
                for dendrita in neurona.dendritas:
                    if dendrita.peso < 0:
                        for s in dendrita.sinapsis:
                            s.peso = min(1.0, max(0.0, s.peso * factor))
            else:
                factor = max(0.01, 1.0 + target)
                for dendrita in neurona.dendritas:
                    if dendrita.peso > 0:
                        for s in dendrita.sinapsis:
                            s.peso = min(1.0, max(0.0, s.peso * factor))

    def balancear_sinapsis(
        self,
        neuronas: list[Neurona],
        target: float = 0.0,
    ) -> None:
        """Remove random synapses from dendrites to shift the balance.

        target > 0: removes synapses from inhibitory dendrites
        target < 0: removes synapses from excitatory dendrites
        target = 0: does nothing

        Each affected dendrite loses floor(len(synapses) * |target|) synapses,
        but always keeps at least 1.
        """
        if target == 0.0:
            return

        factor = abs(target)

        for neurona in neuronas:
            if isinstance(neurona, NeuronaEntrada):
                continue
            if not neurona.dendritas:
                continue

            for dendrita in neurona.dendritas:
                if target > 0 and dendrita.peso >= 0:
                    continue
                if target < 0 and dendrita.peso <= 0:
                    continue

                n_total = len(dendrita.sinapsis)
                if n_total <= 1:
                    continue

                n_remove = int(n_total * factor)
                n_keep = max(1, n_total - n_remove)

                dendrita.sinapsis = random.sample(dendrita.sinapsis, n_keep)

    def aplicar_mascara_2d(
        self,
        red: Red,
        width: int,
        height: int,
        mascara: list[dict[str, object]],
        *,
        random_weights: bool = True,
    ) -> None:
        """Apply a connection mask to every neuron in the 2D grid.

        Args:
            red: The network with the neurons.
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
                neurona_destino = red.get_neurona(self.key_by_coord(x, y))

                for def_dendrita in mascara:
                    peso_dendrita: float = def_dendrita["peso_dendrita"]  # type: ignore[assignment]
                    offsets: list[tuple[int, int]] = def_dendrita["offsets"]  # type: ignore[assignment]
                    pesos_explicitos: list[float] | None = def_dendrita.get("pesos_sinapsis")  # type: ignore[assignment]

                    sinapsis_list: list[Sinapsis] = []
                    for i, (dx, dy) in enumerate(offsets):
                        nx = (x + dx) % width
                        ny = (y + dy) % height
                        neurona_fuente = red.get_neurona(
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

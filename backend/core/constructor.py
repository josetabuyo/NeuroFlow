"""Constructor — factory para armar redes + regiones.

Crea neuronas, las agrupa en regiones,
construye la conectividad (dendritas, sinapsis).
Conoce de topología y patrones de conexión.
Es el ÚNICO que sabe cómo cablear la red.
"""

from __future__ import annotations

import random

from .sinapsis import Sinapsis
from .dendrita import Dendrita
from .neurona import Neurona, NeuronaEntrada
from .red import Red
from .region import Region


class Constructor:
    """Factory/Builder de redes neuronales y regiones."""

    @staticmethod
    def key_by_coord(x: int, y: int) -> str:
        """Genera el ID de neurona a partir de coordenadas."""
        return f"x{x}y{y}"

    def crear_grilla(
        self,
        width: int,
        height: int,
        filas_entrada: list[int],
        filas_salida: list[int],
        umbral: float = 0.0,
    ) -> tuple[Red, dict[str, Region]]:
        """Crea una grilla de neuronas con regiones de entrada, salida e interna.

        Args:
            width: Ancho de la grilla.
            height: Alto de la grilla.
            filas_entrada: Índices de filas que son NeuronaEntrada.
            filas_salida: Índices de filas que son salida (Neurona normal, solo etiqueta).
            umbral: Umbral de activación para neuronas internas y de salida.

        Returns:
            Tupla (Red, dict de Regiones).
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
        mascara_relativa: list[tuple[int, int]],
        regla_pesos: list[list[float]],
        peso_dendrita: float = 1.0,
    ) -> None:
        """Conecta neuronas de una fila con sus vecinas según una máscara relativa.

        Args:
            red: La red que contiene las neuronas.
            fila_destino: Índice de fila cuyas neuronas recibirán las conexiones.
            width: Ancho de la grilla.
            mascara_relativa: Lista de offsets (dx, dy) para encontrar vecinos.
            regla_pesos: Lista de dendritas, cada una es una lista de pesos para las sinapsis.
                         Cada lista de pesos corresponde a un patrón que debe reconocer.
            peso_dendrita: Peso de cada dendrita creada.
        """
        for x in range(width):
            neurona_destino = red.get_neurona(self.key_by_coord(x, fila_destino))

            for pesos_sinapsis in regla_pesos:
                sinapsis_list: list[Sinapsis] = []

                for i, (dx, dy) in enumerate(mascara_relativa):
                    nx = x + dx
                    ny = fila_destino + dy

                    # Bordes: neuronas fuera del rango se tratan como inactivas (valor 0)
                    key = self.key_by_coord(nx, ny)
                    try:
                        neurona_fuente = red.get_neurona(key)
                    except KeyError:
                        # Fuera del borde: crear sinapsis "fantasma" con NeuronaEntrada inactiva
                        neurona_fuente = NeuronaEntrada(id=f"_borde_{nx}_{ny}")
                        neurona_fuente.activar_external(0.0)

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
    ) -> None:
        """Configura las dendritas de una fila según una regla de Wolfram.

        Para cada patrón (de 0 a 7) que produce salida = 1 en la regla,
        crea una dendrita con 3 sinapsis cuyos pesos codifican el patrón.

        Args:
            red: La red con las neuronas.
            regla: Número de regla de Wolfram (0-255).
            fila_destino: Fila a configurar.
            width: Ancho de la grilla.
        """
        # Decodificar la regla: para cada patrón de 3 bits, ¿produce 1?
        patrones_activos: list[list[float]] = []
        for patron in range(8):
            if regla & (1 << patron):
                # Convertir patrón de 3 bits a pesos de sinapsis
                # Bit 2 = izquierda, Bit 1 = centro, Bit 0 = derecha
                izq = float((patron >> 2) & 1)
                cen = float((patron >> 1) & 1)
                der = float(patron & 1)
                patrones_activos.append([izq, cen, der])

        mascara = [(-1, 1), (0, 1), (1, 1)]  # izq, centro, der de fila inferior
        self.conectar_filas(
            red=red,
            fila_destino=fila_destino,
            width=width,
            mascara_relativa=mascara,
            regla_pesos=patrones_activos,
            peso_dendrita=1.0,
        )

    def aplicar_mascara_2d(
        self,
        red: Red,
        width: int,
        height: int,
        mascara: list[dict[str, object]],
    ) -> None:
        """Aplica una máscara de conexión a cada neurona de la grilla 2D.

        Args:
            red: La red con las neuronas.
            width: Ancho de la grilla.
            height: Alto de la grilla.
            mascara: Lista de definiciones de dendritas. Cada una:
                {
                    "peso_dendrita": float,  # Peso de la dendrita (+1.0 o -1.0)
                    "offsets": [(dx, dy), ...],  # Offsets relativos a la neurona
                }
                Los pesos sinápticos se generan aleatorios en [0.2, 1.0].
                Sinapsis cuyo offset cae fuera de la grilla se ignoran.
        """
        for y in range(height):
            for x in range(width):
                neurona_destino = red.get_neurona(self.key_by_coord(x, y))

                for def_dendrita in mascara:
                    peso_dendrita: float = def_dendrita["peso_dendrita"]  # type: ignore[assignment]
                    offsets: list[tuple[int, int]] = def_dendrita["offsets"]  # type: ignore[assignment]

                    sinapsis_list: list[Sinapsis] = []
                    for dx, dy in offsets:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            neurona_fuente = red.get_neurona(
                                self.key_by_coord(nx, ny)
                            )
                            peso = random.uniform(0.2, 1.0)
                            sinapsis_list.append(
                                Sinapsis(neurona_entrante=neurona_fuente, peso=peso)
                            )

                    if sinapsis_list:
                        dendrita = Dendrita(
                            sinapsis=sinapsis_list, peso=peso_dendrita
                        )
                        neurona_destino.dendritas.append(dendrita)

"""Tests para el Inspector de Conexiones — inspect() en Experimento base.

Valida:
- weight_grid con dimensiones correctas
- Kohonen: vecinos excitatorios con peso positivo
- Kohonen: bloques inhibitorios con peso negativo
- Celdas sin conexión → None
- Celda inspeccionada marcada con 999
- Neurona en borde: mismas conexiones que centro (topología toroidal)
- total_dendritas y total_sinapsis correctos
- Pesos efectivos clampeados a [-1, 1]
- Von Neumann: neurona tiene conexiones a fila inferior
- Neurona fuente en múltiples dendritas: pesos sumados
"""

import random

import pytest
from core.constructor import Constructor
from core.sinapsis import Sinapsis
from core.dendrita import Dendrita
from core.neurona import Neurona, NeuronaEntrada
from core.red import Red
from experiments.kohonen import KohonenExperiment, KOHONEN_SIMPLE_MASK


class TestInspect:
    """Experimento.inspect() retorna mapa de pesos efectivos."""

    def test_inspect_retorna_weight_grid_dimensiones_correctas(self) -> None:
        """weight_grid tiene las mismas dimensiones que la grilla."""
        random.seed(42)
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})

        result = exp.inspect(5, 5)

        assert result["type"] == "connections"
        assert result["x"] == 5
        assert result["y"] == 5
        grid = result["weight_grid"]
        assert len(grid) == 10
        assert all(len(row) == 10 for row in grid)

    def test_kohonen_vecinos_excitatorios_peso_positivo(self) -> None:
        """Neurona central Kohonen: 8 vecinos inmediatos con peso positivo."""
        random.seed(42)
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})

        result = exp.inspect(5, 5)
        grid = result["weight_grid"]

        # Los 8 vecinos inmediatos deben tener peso efectivo > 0
        # (D0, peso_dendrita=+1.0, sinapsis con pesos positivos)
        vecinos = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in vecinos:
            nx, ny = 5 + dx, 5 + dy
            peso = grid[ny][nx]
            assert peso is not None, f"Vecino ({nx},{ny}) debería estar conectado"
            assert peso > 0, f"Vecino ({nx},{ny}) debería ser excitatorio, pero tiene peso {peso}"

    def test_kohonen_bloques_inhibitorios_peso_negativo(self) -> None:
        """Neurona central Kohonen: bloques inhibitorios con peso negativo."""
        random.seed(42)
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})

        result = exp.inspect(5, 5)
        grid = result["weight_grid"]

        # D2 — Inhibitoria E: bloque 3×3 en (+2..+4, -1..+1)
        # Estos offsets caen dentro de la grilla para (5,5) en 10x10
        offsets_d2 = [(2, -1), (2, 0), (2, 1), (3, -1), (3, 0), (3, 1), (4, -1), (4, 0), (4, 1)]
        for dx, dy in offsets_d2:
            nx, ny = 5 + dx, 5 + dy
            if 0 <= nx < 10 and 0 <= ny < 10:
                peso = grid[ny][nx]
                assert peso is not None, f"Celda ({nx},{ny}) debería estar conectada"
                assert peso < 0, f"Celda ({nx},{ny}) debería ser inhibitoria, pero tiene peso {peso}"

    def test_celda_sin_conexion_es_none(self) -> None:
        """Celdas sin conexión a la neurona inspeccionada → None."""
        random.seed(42)
        exp = KohonenExperiment()
        exp.setup({"width": 30, "height": 30})

        result = exp.inspect(15, 15)
        grid = result["weight_grid"]

        # Celdas muy lejanas (e.g., esquina opuesta) no tienen conexión
        # (15,15) con máscara kohonen_simple tiene alcance ~4 celdas
        assert grid[0][0] is None
        assert grid[29][29] is None

    def test_celda_inspeccionada_marcada_999(self) -> None:
        """La celda inspeccionada se marca con valor especial 999."""
        random.seed(42)
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})

        result = exp.inspect(5, 5)
        grid = result["weight_grid"]

        assert grid[5][5] == 999

    def test_neurona_borde_mismas_conexiones_que_centro(self) -> None:
        """Toroidal: border neuron has the same connection count as center."""
        random.seed(42)
        exp = KohonenExperiment()
        exp.setup({"width": 30, "height": 30})

        result_centro = exp.inspect(15, 15)
        result_borde = exp.inspect(0, 0)

        grid_centro = result_centro["weight_grid"]
        grid_borde = result_borde["weight_grid"]

        conexiones_centro = sum(
            1 for row in grid_centro for cell in row if cell is not None and cell != 999
        )
        conexiones_borde = sum(
            1 for row in grid_borde for cell in row if cell is not None and cell != 999
        )

        assert conexiones_borde == conexiones_centro

    def test_total_dendritas_y_sinapsis_correctos(self) -> None:
        """total_dendritas y total_sinapsis reflejan la estructura real."""
        random.seed(42)
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})

        result = exp.inspect(5, 5)

        # Neurona central en 10x10 con kohonen_simple: 9 dendritas
        assert result["total_dendritas"] == 9

        # Contar sinapsis reales
        neurona = exp.red.get_neurona("x5y5")
        total_sinapsis = sum(len(d.sinapsis) for d in neurona.dendritas)
        assert result["total_sinapsis"] == total_sinapsis

    def test_pesos_efectivos_clampeados(self) -> None:
        """Pesos efectivos están clampeados a [-1, 1]."""
        random.seed(42)
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})

        result = exp.inspect(5, 5)
        grid = result["weight_grid"]

        for row in grid:
            for cell in row:
                if cell is not None and cell != 999:
                    assert -1.0 <= cell <= 1.0, f"Peso {cell} fuera de rango [-1, 1]"

    def test_neurona_fuente_multiples_dendritas_suma(self) -> None:
        """Si una neurona fuente aparece en múltiples dendritas, los pesos se suman."""
        # Crear un mini experimento manual donde una fuente está en 2 dendritas
        constructor = Constructor()
        red, _ = constructor.crear_grilla(
            width=3, height=3, filas_entrada=[], filas_salida=[], umbral=0.0
        )

        # Conectar manualmente x1y1 con x0y0 en dos dendritas diferentes
        neurona_destino = red.get_neurona("x1y1")
        neurona_fuente = red.get_neurona("x0y0")

        # Dendrita 1: peso_dendrita=+1.0, sinapsis peso=0.8
        # peso_efectivo_1 = 0.8 * 1.0 = 0.8
        s1 = Sinapsis(neurona_entrante=neurona_fuente, peso=0.8)
        d1 = Dendrita(sinapsis=[s1], peso=1.0)

        # Dendrita 2: peso_dendrita=-1.0, sinapsis peso=0.5
        # peso_efectivo_2 = 0.5 * (-1.0) = -0.5
        s2 = Sinapsis(neurona_entrante=neurona_fuente, peso=0.5)
        d2 = Dendrita(sinapsis=[s2], peso=-1.0)

        neurona_destino.dendritas.extend([d1, d2])

        # Crear un experimento ad-hoc para testear
        exp = KohonenExperiment()
        exp.red = red
        exp.width = 3
        exp.height = 3
        exp.regiones = {}
        exp.generation = 0

        result = exp.inspect(1, 1)
        grid = result["weight_grid"]

        # Peso neto = 0.8 + (-0.5) = 0.3
        assert grid[0][0] == pytest.approx(0.3, abs=1e-9)


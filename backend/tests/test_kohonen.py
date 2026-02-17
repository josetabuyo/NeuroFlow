"""Tests para el Experimento Kohonen — competencia lateral 2D (mapa autoorganizado).

Valida:
- aplicar_mascara_2d: conectividad correcta (dendritas, sinapsis, pesos)
- Dinámica excitación/inhibición
- Experimento: setup, click, step, get_frame, reset
"""

import random

import pytest
from core.constructor import Constructor
from core.neurona import Neurona, NeuronaEntrada
from core.red import Red
from experiments.kohonen import KohonenExperiment, KOHONEN_SIMPLE_MASK


class TestAplicarMascara2D:
    """Constructor.aplicar_mascara_2d conecta correctamente la grilla 2D."""

    def test_neurona_central_recibe_9_dendritas(self) -> None:
        """Neurona en el centro de la grilla recibe 9 dendritas (1 exc + 8 inh)."""
        constructor = Constructor()
        red, _ = constructor.crear_grilla(
            width=10, height=10, filas_entrada=[], filas_salida=[], umbral=0.0
        )
        constructor.aplicar_mascara_2d(red, 10, 10, KOHONEN_SIMPLE_MASK)

        # Neurona central (5, 5): lejos de los bordes, recibe todas las dendritas
        neurona = red.get_neurona("x5y5")
        assert len(neurona.dendritas) == 9

    def test_dendrita_excitatoria_tiene_8_sinapsis(self) -> None:
        """La dendrita excitatoria (D0) tiene 8 sinapsis (vecinos inmediatos)."""
        constructor = Constructor()
        red, _ = constructor.crear_grilla(
            width=10, height=10, filas_entrada=[], filas_salida=[], umbral=0.0
        )
        constructor.aplicar_mascara_2d(red, 10, 10, KOHONEN_SIMPLE_MASK)

        neurona = red.get_neurona("x5y5")
        dendrita_exc = neurona.dendritas[0]
        assert dendrita_exc.peso == 1.0
        assert len(dendrita_exc.sinapsis) == 8

    def test_dendrita_inhibitoria_tiene_9_sinapsis(self) -> None:
        """Cada dendrita inhibitoria (D1-D8) tiene 9 sinapsis (bloque 3x3)."""
        constructor = Constructor()
        red, _ = constructor.crear_grilla(
            width=10, height=10, filas_entrada=[], filas_salida=[], umbral=0.0
        )
        constructor.aplicar_mascara_2d(red, 10, 10, KOHONEN_SIMPLE_MASK)

        neurona = red.get_neurona("x5y5")
        for i in range(1, 9):
            dendrita_inh = neurona.dendritas[i]
            assert dendrita_inh.peso == -1.0
            assert len(dendrita_inh.sinapsis) == 9

    def test_neuronas_borde_menos_sinapsis(self) -> None:
        """Neuronas del borde tienen menos sinapsis (vecinos fuera del grid ignorados)."""
        constructor = Constructor()
        red, _ = constructor.crear_grilla(
            width=10, height=10, filas_entrada=[], filas_salida=[], umbral=0.0
        )
        constructor.aplicar_mascara_2d(red, 10, 10, KOHONEN_SIMPLE_MASK)

        # Esquina (0, 0): la dendrita excitatoria pierde vecinos fuera del borde
        neurona_esquina = red.get_neurona("x0y0")
        dendrita_exc = neurona_esquina.dendritas[0]
        # Solo tiene 3 vecinos válidos: (1,0), (0,1), (1,1)
        assert len(dendrita_exc.sinapsis) == 3

    def test_pesos_dendriticos_correctos(self) -> None:
        """D0 tiene peso +1.0, D1-D8 tienen peso -1.0."""
        constructor = Constructor()
        red, _ = constructor.crear_grilla(
            width=10, height=10, filas_entrada=[], filas_salida=[], umbral=0.0
        )
        constructor.aplicar_mascara_2d(red, 10, 10, KOHONEN_SIMPLE_MASK)

        neurona = red.get_neurona("x5y5")
        assert neurona.dendritas[0].peso == 1.0
        for i in range(1, 9):
            assert neurona.dendritas[i].peso == -1.0

    def test_pesos_sinapticos_en_rango(self) -> None:
        """Pesos sinápticos están en [0.2, 1.0]."""
        random.seed(42)
        constructor = Constructor()
        red, _ = constructor.crear_grilla(
            width=10, height=10, filas_entrada=[], filas_salida=[], umbral=0.0
        )
        constructor.aplicar_mascara_2d(red, 10, 10, KOHONEN_SIMPLE_MASK)

        neurona = red.get_neurona("x5y5")
        for dendrita in neurona.dendritas:
            for sinapsis in dendrita.sinapsis:
                assert 0.2 <= sinapsis.peso <= 1.0


class TestDinamicaKohonen:
    """Dinámica de excitación local + inhibición lateral."""

    def test_vecinos_activos_sin_inhibicion_activa_neurona(self) -> None:
        """Neurona con vecinos activos y sin inhibición → tensión positiva → se activa."""
        # Crear una red mínima donde solo hay excitación (sin zonas inhibitorias)
        constructor = Constructor()
        # 3x3 grid: solo excitación (sin inhibición por falta de neuronas lejanas)
        red, _ = constructor.crear_grilla(
            width=3, height=3, filas_entrada=[], filas_salida=[], umbral=0.0
        )
        # Solo aplicar dendrita excitatoria (D0)
        mascara_solo_exc = [KOHONEN_SIMPLE_MASK[0]]
        constructor.aplicar_mascara_2d(red, 3, 3, mascara_solo_exc)

        # Activar todos los vecinos de la neurona central (1,1)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                red.get_neurona(f"x{1+dx}y{1+dy}").activar_external(1.0)

        # Procesar
        neurona_central = red.get_neurona("x1y1")
        neurona_central.procesar()
        neurona_central.activar()

        # Con todos los vecinos activos y pesos aleatorios,
        # la dendrita excitatoria produce un valor positivo
        assert neurona_central.valor == 1.0

    def test_inhibicion_fuerte_desactiva_neurona(self) -> None:
        """Neurona con inhibición fuerte → tensión negativa → no se activa."""
        constructor = Constructor()
        red, _ = constructor.crear_grilla(
            width=10, height=10, filas_entrada=[], filas_salida=[], umbral=0.0
        )

        # Máscara: solo una dendrita inhibitoria con offsets que caigan dentro del grid
        mascara_inh = [
            {
                "peso_dendrita": -1.0,
                "offsets": [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
                            if not (dx == 0 and dy == 0)],
            }
        ]
        constructor.aplicar_mascara_2d(red, 10, 10, mascara_inh)

        # Activar todos los vecinos con valor 1.0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                red.get_neurona(f"x{5+dx}y{5+dy}").activar_external(1.0)

        # Procesar la neurona central
        neurona = red.get_neurona("x5y5")
        neurona.procesar()
        neurona.activar()

        # Con inhibición fuerte, la tensión debería ser negativa
        assert neurona.tension_superficial < 0
        assert neurona.valor == 0.0

    def test_red_pequena_activar_centro_propaga_excitacion(self) -> None:
        """Red 5x5: activar centro → después de 1 step, vecinos se activan."""
        random.seed(42)
        constructor = Constructor()
        red, _ = constructor.crear_grilla(
            width=5, height=5, filas_entrada=[], filas_salida=[], umbral=0.0
        )
        # Solo excitación para probar propagación limpia
        mascara_exc = [KOHONEN_SIMPLE_MASK[0]]
        constructor.aplicar_mascara_2d(red, 5, 5, mascara_exc)

        # Inicializar todo a 0
        for n in red.neuronas:
            n.activar_external(0.0)

        # Activar centro
        red.get_neurona("x2y2").activar_external(1.0)

        # Un step completo
        red.procesar()

        # Al menos algunos vecinos del centro deberían activarse
        vecinos_activos = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n = red.get_neurona(f"x{2+dx}y{2+dy}")
            if n.valor > 0:
                vecinos_activos += 1
        assert vecinos_activos > 0


class TestKohonenExperiment:
    """Experimento Kohonen: orquestación completa."""

    def test_setup_crea_red_30x30(self) -> None:
        """Setup crea red de 30x30 con 900 neuronas."""
        exp = KohonenExperiment()
        exp.setup({"width": 30, "height": 30})
        assert len(exp.red.neuronas) == 900

    def test_todas_neuronas_son_neurona_no_entrada(self) -> None:
        """Todas las neuronas son Neurona (no NeuronaEntrada)."""
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})
        for n in exp.red.neuronas:
            assert type(n) is Neurona
            assert not isinstance(n, NeuronaEntrada)

    def test_neuronas_inicializadas_con_valores_aleatorios(self) -> None:
        """Después del setup, las neuronas tienen valores aleatorios (no todas 0)."""
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})
        valores = [n.valor for n in exp.red.neuronas]
        # No todos iguales
        assert len(set(valores)) > 1
        # Todos en [0, 1]
        assert all(0.0 <= v <= 1.0 for v in valores)

    def test_click_toggle_0_a_1(self) -> None:
        """Click en celda con valor < 0.5 -> activa (valor=1.0)."""
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})

        # Forzar una neurona a 0 via tensor
        idx = 5 * 10 + 5  # y=5, x=5
        exp.red_tensor.set_valor(idx, 0.0)
        exp.click(5, 5)
        assert exp.red_tensor.valores[idx].item() == 1.0

    def test_click_toggle_1_a_0(self) -> None:
        """Click en celda con valor >= 0.5 -> desactiva (valor=0.0)."""
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})

        # Forzar una neurona a 1 via tensor
        idx = 5 * 10 + 5  # y=5, x=5
        exp.red_tensor.set_valor(idx, 1.0)
        exp.click(5, 5)
        assert exp.red_tensor.valores[idx].item() == 0.0

    def test_step_procesa_toda_la_red(self) -> None:
        """Un step procesa toda la red de golpe (red.procesar())."""
        random.seed(42)
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})

        frame_antes = exp.get_frame()
        result = exp.step()
        frame_despues = exp.get_frame()

        # Después de un step, la grilla debería cambiar
        assert result["type"] == "frame"
        assert result["generation"] == 1
        # La grilla cambia (con valores aleatorios, la dinámica produce cambios)
        assert frame_antes != frame_despues

    def test_get_frame_retorna_grilla_correcta(self) -> None:
        """get_frame retorna grilla con dimensiones correctas."""
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})
        frame = exp.get_frame()
        assert len(frame) == 10
        assert all(len(row) == 10 for row in frame)

    def test_reset_reinicializa(self) -> None:
        """Reset reinicializa con nuevos valores aleatorios."""
        exp = KohonenExperiment()
        exp.setup({"width": 10, "height": 10})

        # Hacer algunos steps
        exp.step()
        exp.step()
        gen_antes = exp.generation

        # Reset
        exp.reset()
        assert exp.generation == 0
        # Después del reset, la grilla tiene valores aleatorios (no todos 0)
        frame = exp.get_frame()
        valores = [cell for row in frame for cell in row]
        assert len(set(valores)) > 1

    def test_nunca_retorna_complete(self) -> None:
        """Kohonen nunca termina — siempre se puede seguir haciendo step."""
        exp = KohonenExperiment()
        exp.setup({"width": 5, "height": 5})

        for _ in range(50):
            result = exp.step()
            assert result.get("state") != "complete"
            assert not exp.is_complete()

    def test_is_complete_siempre_false(self) -> None:
        """is_complete() siempre retorna False."""
        exp = KohonenExperiment()
        exp.setup({"width": 5, "height": 5})
        assert exp.is_complete() is False
        for _ in range(10):
            exp.step()
        assert exp.is_complete() is False

    def test_neurona_central_tiene_9_dendritas(self) -> None:
        """En setup 30x30, neurona central tiene exactamente 9 dendritas."""
        exp = KohonenExperiment()
        exp.setup({"width": 30, "height": 30})
        neurona = exp.red.get_neurona("x15y15")
        assert len(neurona.dendritas) == 9

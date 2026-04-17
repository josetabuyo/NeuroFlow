# Analisis: Problemas de Aprendizaje en el Dynamic SOM

## 1. Diagnostico del problema central

Los Daemons se vuelven estaticos y los pesos sinapticos de entrada permanecen ruidosos porque el sistema tiene **varios problemas compuestos** que impiden la diferenciacion. Los analizo en orden de impacto.

**Estado actual:** Despues de multiples iteraciones de desarrollo (spike adaptation, process modes, configurable weights, JSON editor), el SOM todavia no muestra regiones diferenciadas en el espacio que compartan un patron aprendido. Los daemons se ven como circulos blancos (activacion final) o sombreros mexicanos (tension superficial), pero no hay organizacion topologica en respuesta al input.

---

## 2. Aprendizaje durante el gap entre caracteres

**STATUS: RESUELTO**

Con `noise_inter_char=True` (default original), durante el gap entre caracteres se proyectaba ruido blanco puro y las neuronas aprendian hacia patrones aleatorios. Se hizo configurable `noise_inter_char`: con `False` se salta el gap por completo, pasando directo al siguiente caracter.

La hipotesis de "supresion sacadica" (desactivar `learn()` durante el gap manteniendo el gap activo) se descarto como solucion. La supresion sacadica es un mecanismo especifico del ojo/corteza visual V1 que evita procesar imagenes borrosas durante el movimiento ocular, pero no es un principio general del aprendizaje cerebral. El cerebro aprende continuamente — hipocampo, corteza asociativa, etc. no "apagan" el aprendizaje cada vez que cambia el estimulo. No hay razon para creer que desactivar el aprendizaje durante el gap sea crucial para explicar la falta de convergencia del SOM.

---

## 3. Problema del techo plano

**STATUS: NO RESUELTO — FOCO PRINCIPAL DE INVESTIGACION**

### El problema visto desde el aprendizaje

En un SOM clasico de Kohonen, el aprendizaje solo afecta a la **neurona ganadora** (BMU) y sus vecinas dentro de un radio de vecindad. En nuestra implementacion:

```python
def learn(self, lr):
    source_vals = self.valores[self.indices_fuente]
    tension = self.tensiones[:NR].unsqueeze(1)
    delta = lr * tension * (source_vals - self.pesos_sinapsis)
```

La tension modula el aprendizaje para TODAS las neuronas simultaneamente. Si el paisaje de tension es plano (muchas neuronas con tension similar), todas aprenden lo mismo. Esto causa exactamente lo que observamos: "siempre se pisan los patrones de A y B en todas las neuronas".

### El problema visto desde el daemon

La campana del sombrero mexicano tiene un techo plano. Los pesos sinapticos de entrada se inicializan con `random.uniform(0.2, 1.0)`. Con la funcion sinaptica `1 - |peso - entrada|`:

- Si el peso es ~0.6 (media del rango) y la entrada es 0 o 1, la similitud es ~0.4 en ambos casos.
- Todas las neuronas ven similitudes parecidas al principio.
- La dendrita excitatoria (Moore r=3, 48 vecinos) agrupa muchas sinapsis. El promedio de 48 valores tiene varianza baja, lo que aplana el gradiente.
- Con pesos aleatorios y entradas binarias, `1 - |w - input|` promediado sobre 48 sinapsis converge al mismo valor para todos los centros.
- La tension de cada neurona depende mas de los daemons locales que de la entrada.

Resultado: el input no logra crear gradientes suficientes para diferenciar neuronas. No hay un "ganador claro".

### Literatura relevante

El paper "Self-organizing maps: stationary states, metastability and convergence rate" (Springer, 1993) documenta exactamente este fenomeno: **estados metaestables** donde el mapa queda atrapado. La solucion propuesta es usar funciones de vecindad convexas sobre un rango amplio para evitar estos estados.

### Opciones para resolver

**Opcion A:** Winner-take-all explicito (solo top-K neuronas aprenden). Rompe la filosofia distribuida/conexionista.

**Opcion B:** Aumentar `input_dendrite_weight` para que domine. Riesgo: convulsiones.

**Opcion C:** Inicializacion no uniforme de pesos de entrada. Ayuda al inicio pero no resuelve la dinamica.

**Opcion D (PROPUESTA PRINCIPAL):** Funcion de salida no-lineal sobre la tension superficial. Ver seccion 6.

---

## 4. Neuronas con duracion maxima de activacion (Spike Frequency Adaptation)

**STATUS: IMPLEMENTADO Y PROBADO — RESULTADOS INSUFICIENTES**

### Que se implemento

El mecanismo completo de ON/OFF cycling esta en el codigo:

- `adaptation_enabled` (flag activable/desactivable en caliente)
- `max_active_steps` (pasos maximos activos antes de fatiga)
- `refractory_steps` (pasos de descanso forzado)
- Contadores por neurona: `active_counts`, `refractory_remaining`
- Proteccion de NeuronaEntrada (no se fatigan)
- Configurable via JSON editor en la UI

### Resultados observados

Con spike adaptation activado:

- **Visualmente:** Efecto "lava lamp con daemons". Los daemons se mueven, nacen, mueren, migran. Patron biologico y estetico, muy bello.
- **Funcionalmente:** No se observaron regiones en el espacio que compartan el mismo patron aprendido. No hay convergencia SOM.
- **Conclusion:** La fatiga por si sola genera dinamismo pero NO genera organizacion topologica. Es una condicion necesaria (los daemons deben moverse) pero no suficiente (falta el mecanismo que dirija hacia donde se mueven en funcion del input).

### Base biologica confirmada: Spike Frequency Adaptation (SFA)

- Corrientes de potasio activadas por calcio (I_AHP): cada spike acumula calcio intracelular.
- Corrientes de potasio lentas (I_M, I_Ks): se activan con despolarizacion sostenida.
- Duracion limitada de bursts: neuronas piramidales de capas II/III no sostienen bursts >300 Hz.
- HDRP (2025): "Historical Dynamic Refractory Period" — ajusta dinamicamente la duracion del periodo refractario.

---

## 5. Relacion dendritas lejanas vs cercanas

**STATUS: REQUIERE INFRAESTRUCTURA DE TUNING**

La observacion sobre convulsiones cuando `input_dendrite_weight > deamon_exc_weight` es consistente con la teoria de estabilidad de redes:

- **Dendritas cercanas (daemons):** crean la estructura espacial (sombrero mexicano). Funcion geometrica fija.
- **Dendritas lejanas (input):** traen informacion externa que deberia modular, no dominar, la dinamica local.

### Estado actual de las metricas

La metrica de conteo de daemons:
- En momentos de caos: identifica muchisimos daemons (falsos positivos).
- Con daemons bien definidos (3, 4, 6 circulos blancos): los cuenta correctamente.
- **Conclusion:** La metrica es confiable cuando los daemons estan bien formados, que es precisamente cuando la necesitamos para tuning.

### Camino a seguir

No podemos optimizar manualmente la combinacion de:
- `input_dendrite_weight`
- `deamon_exc_weight` / `deamon_inh_weight`
- `learning_rate`
- `max_active_steps` / `refractory_steps`
- Topologia del mask (E, G, I)

Para hacer tuning sistematico (sklearn, algoritmo genetico) necesitamos:
1. Confiar en las metricas existentes (daemon count, noise metric) — **parcialmente logrado**
2. Tener un protocolo de test reproducible — **ver seccion 8**
3. Tener una funcion de fitness compuesta — **pendiente**
4. Infraestructura de ejecucion batch — **pendiente**

### Pre-entrenamiento de sinapsis cercanas

Idea: inicializar sinapsis del daemon a 0.5 (neutral) y correr N pasos sin input para estabilizar.

Riesgo: si las sinapsis cercanas ya son perfectas, la regla hebiana no las modificara (`source - weight` sera pequeno). El daemon queda rigido.

**Este camino se posterga** hasta tener la funcion de salida de tension (seccion 6) funcionando, porque el pre-entrenamiento solo tiene sentido si la dinamica base ya converge.

---

## 6. Funcion de salida de tension superficial (Sharpening)

**STATUS: IMPLEMENTADO — PENDIENTE DE PRUEBAS**

### El problema central

La tension superficial (salida del procesamiento de dendritas) es un valor continuo en [-1, 1] que se usa para dos cosas:
1. **Activacion:** `tension > umbral` → neurona ON/OFF (binario)
2. **Aprendizaje:** `delta = lr * tension * (source - weight)` (continuo)

El problema: la tension "cruda" es plana. Muchas neuronas tienen tensiones similares (~0.3-0.5), asi que el aprendizaje afecta a todas de forma parecida. No hay un "ganador claro".

### La propuesta: aplicar f(tension) antes de usarla

Transformar la tension cruda con una funcion no-lineal que **amplifica las diferencias** y **crea picos puntiagudos**:

```
tension_raw → f(tension_raw) → tension_shaped
```

Donde `tension_shaped` se usa para:
- Determinar activacion (como antes, pero sobre tension_shaped)
- Modular aprendizaje (delta = lr * tension_shaped * ...)

### Opciones de funcion f

#### A. Power law: f(t) = sign(t) * |t|^n

```python
tension_shaped = tension.sign() * tension.abs().pow(n)
```

- `n = 1.0`: sin efecto (lineal, estado actual)
- `n = 2.0`: amplifica diferencias (0.5 → 0.25, 0.9 → 0.81)
- `n = 3.0`: amplifica mucho (0.5 → 0.125, 0.9 → 0.729)

**Ventaja:** Simple, un solo parametro `tension_exponent`. No requiere conocimiento global.
**Desventaja:** No normaliza. Si todas las tensiones son altas, sigue habiendo techo plano.

#### B. Contrast sigmoid: f(t) = tanh(k * t)

```python
tension_shaped = torch.tanh(k * tension)
```

- `k = 1`: suave (casi lineal)
- `k = 3`: moderado
- `k = 10`: muy agudo, casi step function

**Ventaja:** Satura en [-1, 1], crea transicion abrupta.
**Desventaja:** Aplanamiento en los extremos (si muchas tensiones > k, todas saturan a 1).

#### C. Normalizacion divisiva (biologicamente fundamentada)

```python
local_sum = scatter_mean(tension.abs(), neighborhood_indices)
tension_shaped = tension / (sigma + local_sum)
```

La corteza visual usa normalizacion divisiva como mecanismo de gain control (Carandini & Heeger, 2012). La actividad de cada neurona se divide por la actividad promedio de sus vecinas. Esto:

- Amplifica al ganador local respecto a sus vecinos
- Es invariante a la escala global de activacion
- Preserva la estructura espacial

**Ventaja:** Biologicamente realista, invariante a escala, crea ganadores locales.
**Desventaja:** Requiere definir "vecindario" para la normalizacion (puede reusar el del daemon).

#### D. Local softmax (soft winner-take-all)

```python
# Para cada neurona, softmax con sus vecinos locales
# tension_shaped[i] = exp(tension[i]) / sum(exp(tension[j]) for j in vecinos[i])
```

Equivalente a un WTA suave distribuido. La neurona con mayor tension en su vecindario local recibe la mayor proporcion.

**Ventaja:** Teoria solida (Maass, 2000: WTA puede computar funciones arbitrarias).
**Desventaja:** Computacionalmente mas costoso, requiere iteracion por vecindario.

### Implementacion: Power Law (completada)

Se implemento la power law como primer paso. Configurable via `tension_function` en el JSON:

```python
# En BrainTensor.procesar(), despues de calcular tension:
if self.tension_fn == "pow" and self.tension_fn_param != 1.0:
    tension = tension.sign() * tension.abs().pow(self.tension_fn_param)
```

**Config UI:** `"tension_function": {"pow": 2.0}` — aplica power law con exponente 2.
Sin `tension_function` en el JSON (o `{}`) = sin transformacion (raw).
El parametro es "soft": modificable en caliente sin reconnect.

**Archivos modificados:**
- `backend/core/brain_tensor.py`: atributos `tension_fn`, `tension_fn_param`, aplicacion en `procesar()`
- `backend/core/constructor_tensor.py`: passthrough de parametros
- `backend/experiments/dynamic_som.py`: lectura de config, soft param en `update_config()`

**Pendiente:** Ejecutar las pruebas con los presets configurados (Halves Raw vs pow 2/3/5) y documentar resultados. Si power law no es suficiente, escalar a normalizacion divisiva como paso 2.

### Relacion con la literatura

- **Contrast gain control en V1** (Carandini & Heeger, 2012): La corteza visual usa normalizacion divisiva exactamente para este proposito — amplificar diferencias locales manteniendo estabilidad global.
- **Inhibition V2** (2024): Redes spiking que usan pesos negativos fijos desde la neurona ganadora para suprimir competidores, logrando "enhanced signal contrast".
- **Neural field theory** (Pinotsis et al., 2014): El ratio excitacion/inhibicion y la dispersion espacial de las conexiones horizontales determinan la agudeza de la respuesta.

---

## 7. Patrones de entrada simplificados

**STATUS: IMPLEMENTADO — PENDIENTE DE PRUEBAS**

### El temor

Distinguir A de B en fondo negro puede ser un problema mas dificil de lo esperado para un sistema distribuido sin WTA centralizado. Ambos caracteres:
- Ocupan area similar (~30-40% de pixeles blancos)
- Comparten bordes verticales y horizontales
- La diferencia es sutil (patrones internos)

### Propuesta: Escala de dificultad progresiva

Antes de intentar clasificar letras, necesitamos validar que el sistema **puede** auto-organizarse con patrones trivialmente distintos.

#### Nivel 0: Sin input (solo daemons)
Verificar que los daemons se forman, se estabilizan, compiten. Sin input, sin aprendizaje.
**Criterio de exito:** N daemons estables en el grid (ya logrado).

#### Nivel 1: Mitad superior vs mitad inferior
Dos patrones maximamente distintos:
- Patron A: mitad superior blanca, mitad inferior negra
- Patron B: mitad inferior blanca, mitad superior negra
- Sin ruido, sin gap

**Criterio de exito:** Despues de N iteraciones, al inspeccionar los pesos de entrada de neuronas en distintas regiones del grid, las neuronas cercanas deben tener pesos de entrada similares entre si y distintos de neuronas lejanas.

#### Nivel 2: Barras horizontales vs verticales
- Patron A: 3 barras horizontales
- Patron B: 3 barras verticales
- Ruido leve

#### Nivel 3: Punto en esquinas opuestas
- Patron A: punto blanco 5x5 en esquina superior izquierda
- Patron B: punto blanco 5x5 en esquina inferior derecha

#### Nivel 4: MNIST digitos 0 vs 1
Patrones reales pero facilmente distinguibles. 28x28 pixels, binarizados.
**Ventaja:** Benchmark conocido, hay miles de ejemplos por clase, resultados comparables con la literatura.

#### Nivel 5: Letras A vs B (actual)
El desafio original. Solo intentar cuando los niveles 0-3 esten resueltos.

### Implementacion de patrones sinteticos (completada)

Patrones de nivel 1-3 implementados en `dynamic_som.py` como generacion procedural, sin fonts:

- **HALF_TOP / HALF_BOT**: mitad superior / inferior blanca
- **BARS_H / BARS_V**: 3 barras horizontales / verticales
- **DOT_TL / DOT_BR**: punto 5x5 en esquina superior-izquierda / inferior-derecha

Se activan via `input_text` con keywords separadas por coma (ej: `"HALF_TOP,HALF_BOT"`).
La deteccion es automatica: si todas las keywords son patrones sinteticos, se bypasea el font rendering.

### Implementacion de MNIST

Para MNIST necesitamos:
1. Descargar el dataset (torchvision.datasets.MNIST)
2. Filtrar por clase (ej: solo 0 y 1)
3. Redimensionar a `input_resolution x input_resolution`
4. Binarizar con threshold 0.5
5. Iterar ciclicamente por las imagenes del dataset

Esto seria un nuevo tipo de `input_source` configurable en el JSON:
```json
{
    "input_source": "mnist",
    "mnist_classes": [0, 1],
    "input_resolution": 20
}
```

---

## 8. Protocolo de orquestacion de experimentos

**STATUS: PROPUESTA NUEVA**

### Motivacion

Actualmente el experimento corre un loop infinito: generar frame → procesar → aprender → avanzar. Todos los parametros son estaticos durante la ejecucion (salvo cambios manuales via UI).

Para avanzar necesitamos poder definir **fases de entrenamiento** con parametros distintos. Ejemplos:

1. Fase de warmup: 100 steps sin input, solo daemons, sin aprendizaje
2. Fase de presentacion: activar input, activar aprendizaje con LR alto
3. Fase de refinamiento: reducir LR, activar spike adaptation
4. Fase de evaluacion: desactivar aprendizaje, medir metricas

### Referentes en la industria

| Framework | Formato | Descripcion |
|-----------|---------|-------------|
| BMTK (Allen Institute) | JSON (SONATA) | Config declarativa de red, estimulos, duracion, grabacion |
| SpineML | XML | Experiment layer con lesiones, parametros, timing windows |
| NeuroScript (nspy) | DSL propio | Transpila a Python/PyTorch/Brian2 |
| Mozaik | Python | Framework de workflow con experiment control |

### Propuesta: JSON protocol para NeuroFlow

Un archivo JSON que define una secuencia de fases:

```json
{
    "name": "basic_som_test",
    "description": "Test basico de convergencia SOM con mitades",
    "phases": [
        {
            "name": "warmup",
            "steps": 200,
            "config": {
                "input_text": "",
                "learning": false,
                "spike_adaptation": true,
                "max_active_steps": 5
            }
        },
        {
            "name": "train_fast",
            "steps": 500,
            "config": {
                "input_text": "HALF_TOP,HALF_BOT",
                "learning": true,
                "learning_rate": 0.05,
                "noise_inter_char": false,
                "frames_per_char": 20
            }
        },
        {
            "name": "train_slow",
            "steps": 1000,
            "config": {
                "learning_rate": 0.005,
                "spike_adaptation": true
            }
        },
        {
            "name": "eval",
            "steps": 100,
            "config": {
                "learning": false
            },
            "record_metrics": true
        }
    ]
}
```

### Integracion con el sistema actual

El sistema actual ya soporta `update_config()` para cambiar parametros en caliente. El protocolo de orquestacion seria:

1. Cargar el JSON
2. Para cada fase: llamar `update_config(phase.config)`, luego `step_n(phase.steps)`
3. Opcionalmente grabar metricas al final de cada fase
4. Reportar resultados

Esto puede implementarse como un nuevo endpoint en la API (`/run_protocol`) o como un script standalone que use el experimento directamente.

---

## 9. Investigadores y referencias actualizadas

### SOMs distribuidos sin WTA centralizado
- **Asynchronous Distributed SOMs** (2023, arXiv:2301.08379): Convierte unidades del mapa en agentes autonomos con conocimiento local limitado. Updates en cascada (avalanchas). Escala linealmente. Comparable a SOM centralizado.
- **Virtual Winning Neurons** (MDPI Symmetry, 2025): SOM mejorado con neuronas ganadoras virtuales que mejoran la cobertura del espacio.
- **somap** (Python library, 2023): Biblioteca flexible para SOMs con funciones de vecindad customizables.

### Sharpening y contrast gain control
- **Carandini & Heeger (2012):** Normalization divisiva como computacion canonica cortical.
- **Inhibition V2** (Springer, 2024): Lateral inhibition con pesos negativos fijos → 86% MNIST en SNN unsupervised.
- **MUSIC / Manifold-Aware SOM** (2025, arXiv:2601.13851): Framework de inversion de SOMs con control de distancias a prototipos.

### Spiking SOMs con periodos refractarios
- **Diehl & Cook (2015):** SOMs spiking con STDP, periodos refractarios, e inhibicion lateral.
- **HDRP (2025):** Periodo refractario dinamico basado en historial de actividad.
- **RPLIF (2025):** Threshold dynamics — incrementa umbral post-spike.

### Orquestacion de experimentos
- **BMTK (Allen Institute):** JSON SONATA para definir redes, estimulos, duracion.
- **SpineML:** XML experiment layer con timing windows.
- **Mozaik (NeuralEnsemble):** Workflow integrado para simulaciones a gran escala.

---

## 10. Resumen de estado y prioridades

### Resumen de estado por problema

| # | Problema | Estado | Resultado |
|---|----------|--------|-----------|
| 2 | Aprendizaje durante gap | Resuelto | `noise_inter_char=False` salta el gap. |
| 3 | Techo plano | En prueba | Power law sharpening implementado. Ahora probando con input-only learning. |
| 4 | Max active steps (SFA) | Implementado | "Lava lamp". Dinamismo sin organizacion topologica. |
| 5 | Balance dendrita lejana/cercana | Nueva estrategia — ver abajo | Weights recurrentes fijos, solo entrena input. |
| 6 | Funcion de salida (sharpening) | **Implementado** | `tension_function: {"x": N, "x_pow_2": N, "x_pow_3": N}` composable. Soft param. |
| 7 | Patrones simplificados | **Implementado** | HALF_TOP/BOT, BARS_H/V, DOT_TL/BR via input_text. |
| 8 | Orquestacion de protocolos | Propuesta — pendiente | Infraestructura para tuning sistematico. |

### Nuevas herramientas implementadas (abril 2026)

| Feature | Config | Descripcion |
|---------|--------|-------------|
| Per-type learning rates | `learning.lr_exc`, `lr_inh`, `lr_input` | Multiplicadores sobre `rate` por tipo de dendrita. Permite congelar pesos recurrentes y entrenar solo la dendrita de entrada. |
| Densidad de input | `input.density` (0.0–1.0) | Fraccion de neuronas de input que se conectan a cada neurona de tejido. Sparse → potencial especializacion. |
| Auto-fit glyph | — (siempre activo) | Caracteres llenados al borde del grid de input. `padding` configurable. |

### Nueva estrategia activa: input-only learning

En lugar de entrenar todos los pesos simultaneamente (donde el daemon absorbe el gradiente),
se congela el aprendizaje de las dendritas recurrentes y solo se entrena la dendrita de entrada:

```json
"learning": { "rate": 1.0, "lr_exc": 0.0, "lr_inh": 0.0, "lr_input": 0.01 }
```

**Hipotesis:** Si el daemon ya es estable (Stage 1 probado), los pesos recurrentes no necesitan
entrenamiento. El unico objetivo es que los pesos de input diferencien patrones. Al congelar
exc/inh, se elimina la competencia entre gradientes y el input tiene campo libre para especializarse.

**Criterio de exito:** Despues de N steps con HALF_TOP/HALF_BOT, al inspeccionar los pesos de
input de neuronas en distintas regiones del grid, se deben ver patrones diferenciados:
algunas neuronas con pesos altos en la mitad superior, otras en la mitad inferior.

### Prioridades actuales

#### Prioridad 1: Probar input-only learning con HALF_TOP/HALF_BOT
Correr Dynamic SOM con `lr_exc=0, lr_inh=0, lr_input=0.01` durante varios miles de steps.
Usar la herramienta de inspect para ver los pesos de input de neuronas en distintas zonas.
Buscar diferenciacion topografica.

#### Prioridad 2: Variar input density
Probar `density: 0.25` y `density: 0.1`. La conectividad sparse fuerza a distintas neuronas
a especializarse en distintas partes del input — similar a como V1 tiene campos receptivos locales.

#### Prioridad 3: Normalizacion divisiva (si power law no alcanza)
Si el sharpening polinomial no es suficiente para crear ganadores locales, implementar
normalizacion divisiva (Carandini & Heeger, 2012) como siguiente nivel.

#### Prioridad 4: Protocolo de orquestacion basico
Warmup sin input (N steps, solo daemons) → entrenamiento → evaluacion con learning desactivado.
El sistema ya soporta `update_config()` en caliente, solo falta orquestar la secuencia.

---

## 11. Prompt para continuar en otro chat

```
Estamos en Stage 2 (Dynamic SOM) de NeuroFlow — sistema conexionista
basado en daemons con aprendizaje hebbiano.

Estado actual (abril 2026):
- Infraestructura completa: templates, config history (SQLite), JSON editor,
  WebSocket, inspect panel, tension sharpening polinomial
- Per-type learning rates: lr_exc, lr_inh, lr_input (multiplicadores sobre rate)
- Input density: conectividad sparse configurable
- Auto-fit glyph rendering
- Arquitectura unificada: clase Experiment unica, config JSON opt-in

Experimento activo: Dynamic SOM con HALF_TOP/HALF_BOT
- Solo entrenan las dendritas de entrada (lr_exc=0, lr_inh=0, lr_input=0.01)
- Pesos recurrentes (daemon) congelados
- Hipotesis: el daemon ya estable no necesita reentrenamiento; el input
  tiene campo libre para diferenciarse topograficamente

Lo que falta probar:
1. Correr ~2000-5000 steps con el config actual
2. Usar inspect en neuronas de distintas zonas del grid
3. Ver si los pesos de input muestran diferenciacion (mitad sup vs inf)
4. Probar input.density: 0.25 y 0.1 para ver efecto de sparse connectivity
5. Si no hay diferenciacion: considerar normalizacion divisiva (seccion 6)

Archivos clave:
- backend/experiments/experiment.py — logica del experimento
- backend/core/brain_tensor.py — learn() con lr_exc/lr_inh/lr_input
- backend/core/ascii_renderer.py — render_char con auto-fit
- backend/configs/ascii_som.json — config activo del template Dynamic SOM
- docs/analysis_som_learning_problems.md — analisis completo de problemas
- docs/STAGES.md — roadmap con estado actual
```

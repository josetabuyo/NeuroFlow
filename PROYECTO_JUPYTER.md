# 🧠 Connectionist Neural Automaton (CNA)

**Un Autómata Celular Neuronal para la Búsqueda de Inteligencia Artificial**

> *"Del caracol Aplysia al pez cebra: Un modelo conexionista escalable que unifica memoria, predicción y acción"*

---

## 📖 Índice

1. [Visión Filosófica](#visión-filosófica)
2. [Fundamentos Científicos](#fundamentos-científicos)
3. [¿Por qué un Autómata Celular Neuronal?](#por-qué-un-autómata-celular-neuronal)
4. [Del Teatro Cartesiano a la Consciencia Distribuida](#del-teatro-cartesiano-a-la-consciencia-distribuida)
5. [Arquitectura: Reglas Emergentes vs. Hardcoded](#arquitectura-reglas-emergentes-vs-hardcoded)
6. [Stack Tecnológico](#stack-tecnológico)
7. [Plan de Desarrollo](#plan-de-desarrollo)
8. [Notebook 1: Autómata Celular Base](#notebook-1-autómata-celular-base)
9. [Notebook 2: Mapas Auto-Organizados (Kohonen)](#notebook-2-mapas-auto-organizados-kohonen)
10. [Notebook 3: Memoria Temporal y Predicción (HTM)](#notebook-3-memoria-temporal-y-predicción-htm)
11. [Notebook 4: UI Interactiva y Robótica](#notebook-4-ui-interactiva-y-robótica)
12. [Compartir en la Web](#compartir-en-la-web)
13. [Apéndices](#apéndices)
14. [README.md del Proyecto](#readmemd-del-proyecto)

---

## 🎯 Visión Filosófica

### El Problema de la IA Actual: La Gran Brecha

Hoy existe una **brecha fundamental** en la inteligencia artificial:

```
┌──────────────────────────────────────────────────────────┐
│              LA GRAN BRECHA DE LA IA ACTUAL               │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  [IA DE LENGUAJE]        ❌        [ROBÓTICA MÓVIL]     │
│   GPT, Transformers      GAP       Navegación física     │
│   Predicción palabras             Selección acciones     │
│   Sin embodiment                   Sin memoria espacial  │
│   Sin mundo físico                 Sin pensamiento       │
│                                                           │
│                    ¿Qué falta?                           │
│                                                           │
│     🧠 CEREBRO DE BAJO NIVEL                             │
│        • Memoria espacial distribuida                     │
│        • Predicción de acciones (no palabras)            │
│        • Integración sensorial continua                   │
│        • Self-organization espontánea                     │
│        • Embodiment (cuerpo ↔ cerebro)                   │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### La Solución: Connectionist Neural Automaton

Este proyecto propone un **autómata celular neuronal** que:

#### 1️⃣ **No tiene "Teatro Cartesiano"** (Daniel Dennett)

Daniel Dennett criticó la idea del "teatro cartesiano": la falacia de que existe un lugar central en el cerebro donde un "homúnculo" (pequeño observador) ve la experiencia consciente en una pantalla mental.

**Problema:** Si la consciencia ocurre en un lugar específico, ¿quién observa ese lugar? → Regresión infinita

**Solución del CNA:** 
- **No hay centro de control**
- La "inteligencia" emerge de interacciones distribuidas
- Cada neurona es autónoma, solo conoce sus vecinos
- El comportamiento global surge de reglas locales

```python
# ❌ ANTI-PATRÓN (Teatro Cartesiano)
class Brain:
    def think(self):
        data = self.collect_all_sensory_data()
        decision = self.central_controller.decide(data)  # ← Homúnculo!
        self.execute(decision)

# ✅ PATRÓN CNA (Consciencia Distribuida)
class NeuralAutomaton:
    def step(self):
        for neuron in self.neurons:
            # Cada neurona solo ve sus vecinos locales
            neuron.update(neuron.neighbors)  # ← Regla local
        # El pensamiento EMERGE de las interacciones
```

#### 2️⃣ **Reglas Emergentes** (no hardcoded)

Inspirado en autómatas celulares (Conway, Von Neumann), pero con diferencia crucial:

| Tipo | Reglas | Aprendizaje | Ejemplo |
|------|--------|-------------|---------|
| **Clásico** | Hardcoded | ❌ No | Game of Life: "si 3 vecinos vivos → nacer" |
| **CNA** | En sinapsis | ✅ Sí | Peso sináptico aprende: w += η · pre · post |

```python
# Autómata clásico: Reglas fijas
def conway_rule(cell, neighbors):
    alive_neighbors = sum(neighbors)
    if cell == 1:  # Viva
        return 1 if alive_neighbors in [2, 3] else 0
    else:  # Muerta
        return 1 if alive_neighbors == 3 else 0

# CNA: Reglas aprendidas en pesos sinápticos
class Synapse:
    def __init__(self):
        self.weight = random()  # ← Regla inicial aleatoria
    
    def update(self, pre_value, post_value):
        # Hebbian: Regla emerge del uso
        self.weight += 0.01 * pre_value * post_value
```

**Ventaja:** El CNA puede **aprender** qué reglas funcionan para una tarea específica.

#### 3️⃣ **Memoria Distribuida** (moving patterns)

La memoria no está "almacenada" en un lugar, sino que son **patrones de activación que se mueven** por la matriz neuronal:

```
t=0:  [0 0 1 1 0 0 0 0]  ← Patrón inicial (ej: "vi comida")
      ↓ activación se propaga
t=1:  [0 1 1 1 1 0 0 0]
      ↓
t=2:  [1 1 0 0 1 1 0 0]  ← El patrón "viaja"
      ↓
t=3:  [0 0 0 1 1 1 1 0]  ← Activa neuronas motoras → "moverme"
```

- **Memoria de corto plazo:** Activación sostenida en regiones internas
- **Memoria de largo plazo:** Pesos sinápticos modificados (Hebbian)
- **Memory replay:** Patrones se pueden "reproducir" (como en ratas durmiendo)

#### 4️⃣ **Self-Organizing Maps** (Teuvo Kohonen)

El modelo de Kohonen (SOM) muestra cómo redes neuronales forman **mapas topológicos** espontáneamente:

**Función "Sombrero Mexicano" (Mexican Hat):**

```
     Activación
        ↑
        │      ╱‾‾╲         ← Excitación central
        │     ╱    ╲
    ────┼────┼──────┼────  ← Línea base
        │   ╱        ╲     ← Inhibición lateral
        │  ╱          ╲
        └──────────────→ Distancia
```

- **Centro:** Neurona ganadora + vecinos cercanos se activan (excitación)
- **Periferia:** Vecinos lejanos se inhiben (competencia)
- **Resultado:** Clustering espontáneo de patrones similares

**En el cerebro real:**
- **Corteza auditiva:** Mapas tonotópicos (frecuencias vecinas → neuronas vecinas)
- **Corteza visual:** Mapas retinotópicos (campo visual → topología cortical)
- **Corteza somatosensorial:** Homúnculo de Penfield

**En el CNA:**

```python
class Dendrite:
    def lateral_inhibition(self, neighbors, radius=3):
        # Sombrero mexicano
        for i, neighbor in enumerate(neighbors):
            distance = abs(i - self.position)
            if distance <= radius:
                # Excitación: Kohonen neighborhood
                self.value += 0.1 * neighbor.value * (1 - distance/radius)
            else:
                # Inhibición lateral
                self.value -= 0.05 * neighbor.value
```

#### 5️⃣ **Hierarchical Temporal Memory** (Jeff Hawkins)

En *On Intelligence* (2004), Jeff Hawkins propone que el neocórtex funciona como un **sistema de predicción temporal jerárquica**:

**Principios del HTM:**

1. **Sparse Distributed Representations:** Solo ~2% de neuronas activas simultáneamente
2. **Sequence Learning:** Aprende patrones temporales (A→B→C)
3. **Prediction:** Predice el siguiente estado basado en secuencias aprendidas
4. **Hierarchy:** Niveles superiores aprenden patrones de patrones

```
Nivel 3: [Concepto abstracto: "peligro"]
           ↑
Nivel 2: [Secuencia: "sombra→movimiento→forma"]
           ↑
Nivel 1: [Píxeles: bordes, texturas]
           ↑
Entrada: [Sensor visual]
```

**En el CNA:**

```python
class HTMLayer:
    def predict(self, current_state):
        # Busca secuencias conocidas: A→B→?
        predicted_next = self.sequence_memory.get(current_state)
        
        # Pre-activa neuronas esperadas (predictive state)
        for neuron in predicted_next:
            neuron.tension += 0.5  # Umbral más bajo
        
        # Si la predicción acierta → refuerzo
        # Si falla → sorpresa → aprendizaje
```

#### 6️⃣ **Place Cells & Grid Cells** (O'Keefe, Moser)

Descubrimientos en ratas (Premio Nobel 2014):

- **Place cells** (hipocampo): Neuronas que se activan en lugares específicos
- **Grid cells** (corteza entorrinal): Patrón hexagonal que cubre el espacio

**Hallazgos recientes (2024):**
- Las ratas pueden **imaginar lugares** sin estar allí (memory replay)
- Los patrones de activación "repasan" rutas durante el sueño
- **Predictive grid cells:** Se activan en la posición FUTURA (no solo actual)

```
    Lugar A              Lugar B
      [●]────────────────[●]
       ↑                  ↑
    Place cell 1      Place cell 2

Durante navegación:
  t=0: Place cell 1 activa ✓
  t=1: Ambas activas (transición)
  t=2: Place cell 2 activa ✓

Durante imaginación (sin movimiento):
  t=0: Place cell 1 activa
  t=1: Secuencia se reproduce internamente
  t=2: Place cell 2 activa ← ¡Sin moverse!
```

**En el CNA:**

```python
class SpatialMap:
    def __init__(self):
        self.place_cells = {}  # (x,y) → neuron
        self.grid_cells = []   # Patrón hexagonal
    
    def imagine_path(self, start, goal):
        # Memory replay: Activa secuencia sin input sensorial
        path = self.find_path(start, goal)
        for position in path:
            self.place_cells[position].activate()
            yield position  # ← Predicción
```

#### 7️⃣ **Integración Sensorial** (multimodal)

El cerebro integra información de múltiples fuentes:

```
┌─────────────────────────────────────────┐
│         INTEGRACIÓN SENSORIAL            │
├─────────────────────────────────────────┤
│                                          │
│  [Vista]    [Tacto]    [Oído]          │
│     ↓          ↓         ↓              │
│  Región     Región    Región            │
│  ENTRADA    ENTRADA   ENTRADA           │
│     ╲         │        ╱                │
│      ╲        │       ╱                 │
│       ╲       │      ╱                  │
│        ↓      ↓     ↓                   │
│     Región INTERNA                      │
│   (Neuronas integradoras)               │
│            ↓                             │
│       Región SALIDA                      │
│      (Neuronas motoras)                  │
│            ↓                             │
│        [Acción]                          │
│                                          │
└─────────────────────────────────────────┘
```

**Ejemplo del pez cebra (2024):**
- Imaging de cerebro completo durante comportamiento libre
- Neuronas en el preoptic nucleus integran:
  - Información visual (novedad del tanque)
  - Estado interno (curiosidad vs. miedo)
  - Memoria (lugares ya explorados)
- La activación integrada predice: explorar vs. esconderse

#### 8️⃣ **Acción como Predicción** (no palabras)

La IA actual predice **tokens** (palabras). Este modelo predice **acciones**:

| Modelo | Entrada | Predicción | Objetivo |
|--------|---------|------------|----------|
| **GPT** | "El gato está en el..." | "sofá" | Completar frase |
| **CNA** | [Vista: comida a izquierda] | mover_izquierda() | Supervivencia |

**"Pensar es predecir qué hacer"** (embodied cognition)

```python
class EmbodiedBrain:
    def think(self, sensory_input):
        # 1. Integrar sensores
        state = self.integrate(sensory_input)
        
        # 2. Predecir consecuencias de acciones
        predictions = {
            'move_left': self.predict(state, action='left'),
            'move_right': self.predict(state, action='right'),
            'stay': self.predict(state, action='stay')
        }
        
        # 3. Seleccionar acción con mejor predicción
        best_action = max(predictions, key=lambda a: predictions[a].value)
        
        return best_action
```

#### 9️⃣ **Escalabilidad: De Aplysia a Pez Cebra**

**Aplysia californica (caracol marino):**
- 20,000 neuronas (vs. 86 mil millones en humanos)
- Eric Kandel (Nobel 2000) descubrió mecanismos de **memoria y aprendizaje**
- Reflejo de retracción branquial: sensibilización y habituación
- **Lección:** Mecanismos básicos (Hebbian, LTP) son universales

**Progresión:**

```
Aplysia       C. elegans      Mosca       Pez cebra      Rata         Humano
(20K)         (302)           (100K)      (100M)         (200M)       (86B)
   ↓             ↓              ↓            ↓             ↓            ↓
Reflejos → Quimiotaxis → Navegación → Mapas → Planificación → Lenguaje
```

**Estrategia del CNA:**

1. **Fase 1:** Implementar modelo de Aplysia (reflejos condicionados)
2. **Fase 2:** Añadir navegación espacial (pez cebra, place cells)
3. **Fase 3:** Integrar HTM para memoria temporal (rata)
4. **Fase 4:** Conectar con transformers para procesamiento simbólico

#### 🔟 **Conexión con Transformers** (lo mejor de ambos mundos)

```
┌────────────────────────────────────────────────────────────┐
│              ARQUITECTURA HÍBRIDA CNA+TRANSFORMERS           │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  [NIVEL ALTO: Transformers]                                │
│   • Embeddings de lenguaje                                 │
│   • Razonamiento simbólico                                 │
│   • Atención global                                        │
│   • "Qué hacer en esta situación"                         │
│            ↕                                               │
│  [INTERFAZ: Embedding <-> Activación]                      │
│   • Traducir palabras → patrones neuronales               │
│   • Traducir activaciones → acciones simbólicas           │
│            ↕                                               │
│  [NIVEL BAJO: CNA]                                         │
│   • Autómata celular neuronal                             │
│   • Memoria espacial distribuida                           │
│   • Predicción de acciones físicas                         │
│   • "Cómo ejecutar la acción"                             │
│            ↕                                               │
│  [CUERPO: Sensores + Actuadores]                           │
│   • Vista, tacto, propriocepción                          │
│   • Motores, músculos                                      │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**Ejemplo de integración:**

```python
# Transformer de alto nivel decide QUÉ hacer
instruction = gpt_model("Estoy viendo comida a la izquierda")
# Output: "move_to_food"

# CNA traduce a activación neuronal
embedding = embed(instruction)  # [0.2, 0.8, -0.3, ...]
motor_region.set_pattern(embedding)

# CNA ejecuta CÓMO hacerlo
for t in range(100):
    cna.step()  # Autómata neuronal genera secuencia motora
    robot.apply_forces(cna.motor_output)
```

---

## 🔬 Fundamentos Científicos

### Estudios que validan este enfoque

#### 1. **Zebrafish Whole-Brain Imaging (2024-2025)**

**Paper:** "Whole-brain mapping in adult zebrafish and identification of a novel tank test functional connectome"

**Hallazgos:**
- Imaging del cerebro completo durante comportamiento libre
- Técnica: Light-sheet microscopy + machine learning
- El **preoptic nucleus anterior** actúa como hub integrando:
  - Telencéfalo ventral (emoción)
  - Regiones sensoriales (vista)
  - Núcleos de neurotransmisores (dopamina, serotonina)

**Relevancia para CNA:**
- Valida el modelo de **regiones integradoras**
- Las neuronas internas no tienen función predefinida, sino que emergen como hubs
- La activación se propaga espacialmente (como en autómata celular)

#### 2. **Predictive Grid Cells (2024)**

**Paper:** "Mapping future locations" (Nature Neuroscience)

**Hallazgos:**
- Grid cells en corteza entorrinal **predicen posición futura**
- No solo codifican "dónde estoy", sino "dónde estaré"
- Durante imaginación, las secuencias se activan sin movimiento

**Relevancia para CNA:**
- La predicción es fundamental (HTM correcto)
- Memory replay = activación interna sin input
- El cerebro es una "máquina de predecir acciones"

#### 3. **Neural Cellular Automata (2020)**

**Paper:** "Growing Neural Cellular Automata" (Distill.pub, Google Research)

**Hallazgos:**
- Reemplazar reglas fijas de autómatas con **redes neuronales entrenables**
- Cada célula ejecuta la misma regla (red neuronal)
- Emergen propiedades: morfogénesis, regeneración, auto-reparación

**Relevancia para CNA:**
- Demuestra que **autómatas + aprendizaje = comportamiento emergente complejo**
- Confirma que reglas locales + gradiente → inteligencia global

#### 4. **Hierarchical Temporal Memory - Numenta**

**Paper:** "A Framework for Intelligence and Cortical Function Based on Grid Cells in the Neocortex" (Hawkins et al.)

**Hallazgos:**
- El neocórtex usa **columnas corticales** como unidades repetitivas
- Cada columna aprende secuencias temporales
- Niveles superiores aprenden patrones de patrones

**Relevancia para CNA:**
- Arquitectura jerárquica es clave para escalabilidad
- Predicción temporal debe ser explícita en el modelo

#### 5. **Kohonen Self-Organizing Maps (1990)**

**Paper:** "The self-organizing map" (Teuvo Kohonen, Proceedings of the IEEE)

**Hallazgos:**
- Mapas topológicos emergen de aprendizaje competitivo
- Función de vecindad (sombrero mexicano) crucial
- Similares a mapas corticales reales (visual, auditivo, somatosensorial)

**Relevancia para CNA:**
- **Dendritas laterales de excitación/inhibición** implementan esto
- Clustering espontáneo sin supervisión

---

## ❓ ¿Por qué un Autómata Celular Neuronal?

### Comparación de paradigmas

| Aspecto | Red Neuronal Clásica | Transformer | **Autómata Celular Neuronal** |
|---------|---------------------|-------------|-------------------------------|
| **Conectividad** | Feed-forward fija | All-to-all attention | Vecindad local dinámica |
| **Temporalidad** | Implícita (RNN) | Posicional encoding | Explícita (estado evoluciona) |
| **Memoria** | Pesos estáticos | Context window | Patrones dinámicos distribuidos |
| **Espacialidad** | No explícita | No explícita | ✅ Topología 2D/3D nativa |
| **Aprendizaje** | Backprop global | Backprop global | Hebbian local + Backprop |
| **Escalabilidad** | O(N²) conexiones | O(N²) atención | O(N) vecindad constante |
| **Interpretabilidad** | Baja | Muy baja | ✅ Alta (ver patrones evolucionar) |
| **Embodiment** | No | No | ✅ Sí (topología = espacio físico) |

### Ventajas del CNA

#### 1. **Localidad espacial explícita**

```python
# Red feed-forward: Neurona 5 puede conectar con cualquiera
layer[5].forward([w1*n0, w2*n1, w3*n2, ..., w100*n99])  # ← No locality

# CNA: Neurona (x,y) solo ve vecinos 3x3
grid[x][y].forward([
    grid[x-1][y-1], grid[x][y-1], grid[x+1][y-1],  # Arriba
    grid[x-1][y],   grid[x][y],   grid[x+1][y],    # Centro
    grid[x-1][y+1], grid[x][y+1], grid[x+1][y+1]   # Abajo
])  # ← Solo 9 vecinos, no 100
```

**Beneficio:** 
- Menos parámetros (O(N) vs O(N²))
- Más eficiente en GPU (convoluciones)
- Mapas espaciales emergen naturalmente

#### 2. **Evolución temporal visible**

```python
# Transformer: Un forward pass opaco
output = transformer(input)  # ¿Qué pasó aquí? 🤷

# CNA: Puedes ver cada paso
for t in range(100):
    cna.step()
    visualize(cna.grid)  # ← Ver patrones moverse, emerger, colapsar
```

**Beneficio:**
- Debugging intuitivo
- Experimentación interactiva
- Educativo (ver "pensamiento" en tiempo real)

#### 3. **Memoria distribuida persistente**

```python
# RNN: Memoria en hidden state (se desvanece)
h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t)  # ← Gradientes exploding/vanishing

# Transformer: Memoria en context window (limitado)
output = attention(query, keys[:max_length])  # ← Solo últimos N tokens

# CNA: Memoria en activación + pesos (persistente)
grid[x][y].value = 0.8  # ← Permanece hasta que algo la cambie
synapse.weight += hebbian_update()  # ← Memoria a largo plazo
```

**Beneficio:**
- No hay "olvido catastrófico"
- Memoria de corto plazo (activación) + largo plazo (pesos)
- Memory replay factible

#### 4. **Reglas emergentes, no diseñadas**

```python
# Algoritmo A*: Reglas de búsqueda hardcoded
def a_star(start, goal):
    open_set = {start}
    while open_set:
        current = min(open_set, key=lambda n: f_score[n])  # ← Regla fija
        # ...

# CNA: Reglas emergen del aprendizaje
# Entrenas con muchos ejemplos (start, goal) → path
# La red aprende a propagar activación en la dirección correcta
```

**Beneficio:**
- Generalización a nuevos escenarios
- No necesitas programar todas las reglas
- Adapta las reglas si el entorno cambia

#### 5. **Unifica percepción y acción**

```
    ┌─────────────────────────────────────────┐
    │         ARQUITECTURA CLÁSICA            │
    ├─────────────────────────────────────────┤
    │  [Percepción] → [Razonamiento] → [Acción] │
    │      CNN          MLP            Policy   │
    │   (separado)   (separado)      (separado) │
    └─────────────────────────────────────────┘

    ┌─────────────────────────────────────────┐
    │         ARQUITECTURA CNA                │
    ├─────────────────────────────────────────┤
    │             [ÚNICA MATRIZ]              │
    │  Región ENTRADA → Región INTERNA → Región SALIDA │
    │  (píxeles)        (procesamiento)   (motores)     │
    │  Todo es el mismo autómata evolutivo    │
    └─────────────────────────────────────────┘
```

**Beneficio:**
- Feedback sensorimotor directo
- Aprendizaje end-to-end natural
- Embodiment intrínseco

---

## 🎭 Del Teatro Cartesiano a la Consciencia Distribuida

### El Teatro Cartesiano (lo que NO queremos)

René Descartes imaginó la glándula pineal como el punto donde mente (res cogitans) y cuerpo (res extensa) interactuaban.

Daniel Dennett critica esta idea y su versión moderna materialista:

```
    ┌────────────────────────────────────┐
    │      TEATRO CARTESIANO             │
    ├────────────────────────────────────┤
    │                                    │
    │  [Ojos] ─────→ [Imagen en retina] │
    │                      ↓             │
    │  [Oídos] ────→ [Corteza visual]   │
    │                      ↓             │
    │  [Tacto] ────→ [Procesamiento]    │
    │                      ↓             │
    │              ┏━━━━━━━━━━━┓        │
    │              ┃  PANTALLA  ┃        │
    │              ┃    MENTAL  ┃        │
    │              ┗━━━━━━━━━━━┛        │
    │                    ↑               │
    │              [HOMÚNCULO]           │
    │           (¿quién mira?)           │
    │                  ↓                 │
    │              [DECISIÓN]            │
    │                                    │
    └────────────────────────────────────┘
         
         ❌ Problema: Regresión infinita
         ¿Quién observa al homúnculo?
```

### Consciencia Distribuida (lo que SÍ queremos)

**Modelo "Multiple Drafts" de Dennett:**

- No hay un lugar donde "todo se junta"
- Múltiples procesos paralelos compiten y colaboran
- La experiencia consciente es un producto emergente
- No hay un "momento de presentación" único

```
    ┌────────────────────────────────────────────────┐
    │         CONSCIENCIA DISTRIBUIDA                │
    ├────────────────────────────────────────────────┤
    │                                                │
    │  [Ojos] ──→ [V1] ──→ [V2] ──→ [V4] ──→ [IT]  │
    │               ↓       ↓       ↓       ↓       │
    │  [Oídos] ──→ [A1] ──→ [A2] ──→ [Integración] │
    │               ↓       ↓           ↓    ↓      │
    │  [Tacto] ──→ [S1] ──→ [Parietal] ↓    ↓      │
    │               ↓           ↓       ↓    ↓      │
    │             [Memoria] ←→ [Predicción]  ↓      │
    │                 ↓           ↓         ↓       │
    │               [Motor] ←───────────────↓       │
    │                                                │
    │   ✅ No hay centro                            │
    │   ✅ Todo interactúa con todo                 │
    │   ✅ La "consciencia" EMERGE                  │
    │                                                │
    └────────────────────────────────────────────────┘
```

### Implementación en el CNA

```python
class ConnessionistNeuralAutomaton:
    def __init__(self):
        # No hay "control central"
        self.neurons = self.create_grid(128, 128)
        
        # Cada neurona es autónoma
        for neuron in self.neurons:
            neuron.autonomous = True  # ← No espera órdenes
    
    def step(self):
        # ❌ NO HACER (centralizado):
        # self.central_controller.decide()
        # for neuron in self.neurons:
        #     neuron.value = self.central_controller.outputs[neuron.id]
        
        # ✅ SÍ HACER (distribuido):
        for neuron in self.neurons:
            # Cada neurona decide basada SOLO en sus vecinos
            neuron.update_from_neighbors()
        
        # El comportamiento global EMERGE
        # No hay "quien decide" globalmente
```

**Comparación con el proyecto original:**

> *"No hay operaciones que se hagan por fuera de las conexiones, el modelo es conexionista puro, no hay un espectador dentro del cerebro regulando la actividad, puede distribuirse sin problemas."*

✅ **Esto es exactamente Dennett**: Sin teatro, sin espectador, sin homúnculo.

---

## ⚙️ Arquitectura: Reglas Emergentes vs. Hardcoded

### Autómatas Celulares Clásicos

#### Game of Life (John Conway)

Reglas fijas:

```python
def game_of_life(cell, neighbors):
    alive_neighbors = sum(neighbors)
    
    if cell == 1:  # Célula viva
        if alive_neighbors < 2:
            return 0  # Muerte por soledad
        elif alive_neighbors in [2, 3]:
            return 1  # Supervivencia
        else:
            return 0  # Muerte por sobrepoblación
    else:  # Célula muerta
        if alive_neighbors == 3:
            return 1  # Nacimiento
        else:
            return 0  # Sigue muerta
```

**Comportamiento emergente:**
- Gliders (planeadores que se mueven)
- Oscillators (osciladores periódicos)
- Still lifes (patrones estáticos)

**Limitación:** Las reglas son fijas. No puede "aprender" qué reglas funcionan mejor.

#### Autómata de Von Neumann

Más complejo: 29 estados por célula, reglas para auto-replicación.

**Logro:** Demostró que autómatas pueden replicarse (concepto de vida artificial).

**Limitación:** Reglas diseñadas manualmente, no aprendidas.

### Neural Cellular Automata (Mordvintsev et al., 2020)

**Innovación:** Reemplazar reglas fijas con **redes neuronales diferenciables**:

```python
class NeuralCA(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        # La "regla" es una red neuronal
        self.network = nn.Sequential(
            nn.Conv2d(channels*3, 128, 1),  # 3x3 vecindad
            nn.ReLU(),
            nn.Conv2d(128, channels, 1)
        )
    
    def forward(self, grid):
        # Percibir vecindad (3x3 convolution)
        perception = self.perceive(grid)
        
        # Aplicar regla aprendida
        update = self.network(perception)
        
        # Actualizar con residual connection
        return grid + update * 0.1
    
    def perceive(self, grid):
        # 3x3 Sobel filters para detectar gradientes
        return torch.cat([
            F.conv2d(grid, self.sobel_x),
            F.conv2d(grid, self.sobel_y),
            grid
        ], dim=1)
```

**Entrenamiento:**

```python
# Objetivo: Crecer un emoji 🦎 desde una semilla
target_image = load_emoji("🦎")
seed = torch.zeros_like(target_image)
seed[0, height//2, width//2] = 1.0  # Semilla central

# Entrenar
optimizer = torch.optim.Adam(nca.parameters())
for epoch in range(1000):
    # Simular evolución
    state = seed.clone()
    for t in range(64):  # 64 pasos de autómata
        state = nca(state)
    
    # Loss: ¿Se parece al emoji objetivo?
    loss = F.mse_loss(state, target_image)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Resultado:** La red aprende reglas que hacen crecer el emoji. Bonus: regenera si se daña.

### Connectionist Neural Automaton (nuestro modelo)

**Diferencia con NCA clásico:**

| Aspecto | NCA (Mordvintsev) | **CNA (nuestro)** |
|---------|-------------------|-------------------|
| Objetivo | Morfogénesis (crecer imagen) | **Cognición** (pensar, actuar) |
| Regla | Una red neuronal global | **Sinapsis individuales** (más granular) |
| Aprendizaje | Backprop supervisado | **Hebbian + Backprop híbrido** |
| Estructura | Grid homogéneo | **Regiones** (ENTRADA/INTERNA/SALIDA) |
| Biología | Inspirado en desarrollo | **Inspirado en cerebro funcional** |

**Arquitectura del CNA:**

```python
class CNA:
    def __init__(self, width=64, height=64):
        # Matriz neuronal 2D
        self.grid = [[Neuron(x, y) for x in range(width)] for y in range(height)]
        
        # Regiones funcionales (como en el cerebro)
        self.regions = {
            'ENTRADA': self.grid[0:16],      # Sensores (arriba)
            'INTERNA': self.grid[16:48],     # Procesamiento (centro)
            'SALIDA':  self.grid[48:64],     # Motores (abajo)
            'DOLOR':   self.grid[60:64]      # Señal de error
        }
        
        # Conectar neuronas
        for neuron in self.all_neurons():
            neuron.dendrites = [
                Dendrite([
                    Synapse(neighbor, weight=random())
                    for neighbor in self.get_neighbors(neuron, radius=3)
                ])
                for _ in range(4)  # 4 dendritas por neurona
            ]
    
    def step(self):
        # 1. Procesar dendritas (AND difuso)
        for neuron in self.all_neurons():
            for dendrite in neuron.dendrites:
                dendrite.procesar()  # Promedio de sinapsis
        
        # 2. Procesar neuronas (OR difuso + activación)
        for neuron in self.all_neurons():
            neuron.procesar()  # Max de dendritas
            neuron.activar()   # Umbral
        
        # 3. Entrenar sinapsis (Hebbian)
        if self.learning_enabled:
            for neuron in self.all_neurons():
                for dendrite in neuron.dendrites:
                    dendrite.entrenar()  # w += η·pre·post

class Synapse:
    def __init__(self, source_neuron, weight=0.5):
        self.source = source_neuron
        self.weight = weight  # ← AQUÍ ESTÁ LA REGLA
    
    def procesar(self):
        # Similaridad entre neuronas
        pre = self.source.valor
        post = self.target.valor
        return self.weight * (1 - abs(pre - post))
    
    def entrenar(self):
        # Hebbian: "Neurons that fire together, wire together"
        pre = self.source.valor
        post = self.target.valor
        
        self.weight += 0.01 * pre * post  # ← REGLA APRENDIDA
        
        # Poda sináptica
        if self.weight < 0.1:
            self.weight = 0  # Eliminar sinapsis débil
```

**Ventaja:** Cada sinapsis tiene su propia "regla" (peso), y todas aprenden en paralelo.

### Comparación: Reglas Hardcoded vs. Emergentes

```
┌──────────────────────────────────────────────────────────┐
│                 REGLAS HARDCODED                         │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  if (neighbors == 3) { alive = true; }                  │
│  else if (neighbors < 2) { alive = false; }             │
│                                                           │
│  ✅ Pros:                                               │
│     • Simple de entender                                 │
│     • Determinista                                       │
│                                                           │
│  ❌ Contras:                                            │
│     • No se adapta a nuevas situaciones                 │
│     • Diseñador debe saber las reglas correctas         │
│     • No generaliza                                      │
│                                                           │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                 REGLAS EMERGENTES (CNA)                  │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  weight += learning_rate * pre_activation * post_activation │
│  output = sum(weight[i] * neighbor[i])                  │
│                                                           │
│  ✅ Pros:                                               │
│     • Aprende reglas óptimas para la tarea              │
│     • Se adapta si el entorno cambia                    │
│     • Descubre soluciones no obvias                     │
│     • Generaliza a situaciones nuevas                   │
│                                                           │
│  ❌ Contras:                                            │
│     • Necesita datos de entrenamiento                   │
│     • Menos predecible                                   │
│     • Puede converger a óptimos locales                 │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

**Ejemplo concreto:**

**Tarea:** Navegar un robot hacia comida.

```python
# ❌ Reglas hardcoded
def navigate(sensors):
    if sensors['left'] > sensors['right']:
        return 'turn_left'
    elif sensors['right'] > sensors['left']:
        return 'turn_right'
    else:
        return 'forward'

# ✅ Reglas emergentes (CNA)
# Entrenar con 1000 ejemplos de (sensores, acción correcta)
for episode in range(1000):
    state = env.reset()
    for t in range(100):
        # Poner sensores en región ENTRADA
        cna.set_input_region(state)
        
        # Propagar activación
        for _ in range(10):
            cna.step()
        
        # Leer acción de región SALIDA
        action = cna.get_output_region()
        
        # Ejecutar
        next_state, reward = env.step(action)
        
        # Entrenar (Hebbian + refuerzo)
        if reward > 0:
            cna.reinforce_active_synapses()  # ← Reforzar pesos actuales
        else:
            cna.weaken_active_synapses()
```

Después de entrenamiento, el CNA ha aprendido reglas como:

- "Si neurona de sensor izquierdo activa → neurona motora izquierda activa"
- "Si ambos sensores activos → neurona motora frontal activa"
- Y muchas reglas sutiles difíciles de programar manualmente

---

## 🛠️ Stack Tecnológico

### Lenguaje y Entorno

- **Python 3.11+**: Lenguaje principal
- **Jupyter Notebook**: Entorno interactivo
- **Google Colab**: Hosting gratuito con GPU (T4, P100)
- **Binder/MyBinder**: Alternativa open-source

### Computación Neuronal

| Librería | Propósito | ¿Por qué? |
|----------|-----------|-----------|
| **PyTorch** | Framework de deep learning | • Dinámico (vs. TensorFlow estático)<br>• Excelente para investigación<br>• `torch.compile()` para optimización<br>• Soporte nativo de GPU |
| **NumPy** | Operaciones matriciales | • Base de todo<br>• Integración perfecta con PyTorch |

### Transformers (Nivel Alto)

| Librería | Propósito | ¿Por qué? |
|----------|-----------|-----------|
| **Transformers (Hugging Face)** | Modelos pre-entrenados | • Embeddings de lenguaje<br>• BERT, GPT, etc.<br>• Fácil integración |
| **Sentence-Transformers** | Embeddings semánticos | • Texto → Vector denso<br>• Comparación de significado |

### Visualización

| Librería | Propósito | ¿Por qué? |
|----------|-----------|-----------|
| **ipycanvas** | Canvas interactivo | • Dibujar píxeles en tiempo real<br>• Eventos de mouse<br>• Renderizado eficiente |
| **ipywidgets** | Controles UI | • Sliders, botones, dropdowns<br>• Interactividad sin JavaScript |
| **Matplotlib** | Gráficos 2D | • Plots científicos<br>• Heatmaps, evolución temporal |
| **Plotly** | Gráficos interactivos | • 3D opcional<br>• Zoom, hover |

### Optimización

```python
# torch.compile() - PyTorch 2.0+
@torch.compile(mode="reduce-overhead")
def cna_step(states, weights):
    # 100x más rápido que código Python puro
    return F.conv2d(states, weights)

# Mixed Precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    output = model(input)  # FP16/BF16 automático

# Flash Attention (para transformers grandes)
from torch.nn.functional import scaled_dot_product_attention
attn = scaled_dot_product_attention(q, k, v)  # 2-4x más rápido
```

### Estructura de Archivos

```
CNA_Project/
├── README.md                    # ← Descripción del proyecto
├── requirements.txt             # ← pip install -r requirements.txt
├── environment.yml              # ← conda env create -f environment.yml
├── notebooks/
│   ├── 01_Automata_Base.ipynb
│   ├── 02_SOM_Kohonen.ipynb
│   ├── 03_HTM_Prediccion.ipynb
│   └── 04_UI_Robotica.ipynb
├── src/
│   ├── cna/
│   │   ├── __init__.py
│   │   ├── core.py              # Neurona, Dendrita, Sinapsis
│   │   ├── automaton.py         # CNA main class
│   │   ├── regions.py           # Región ENTRADA/SALIDA/INTERNA
│   │   ├── learning.py          # Hebbian, STDP
│   │   └── visualization.py     # Rendering
│   ├── som/
│   │   ├── kohonen.py           # Self-Organizing Map
│   │   └── mexican_hat.py       # Lateral inhibition
│   ├── htm/
│   │   ├── temporal_memory.py   # Hawkins HTM
│   │   └── spatial_pooler.py
│   └── transformers/
│       ├── embeddings.py        # Interfaz con HuggingFace
│       └── hybrid.py            # CNA + Transformer
├── experiments/
│   ├── aplysia_reflex.py        # Reflejo condicionado simple
│   ├── zebrafish_navigation.py  # Navegación espacial
│   └── rat_memory_replay.py     # Memory replay
├── assets/
│   ├── diagrams/                # Diagramas explicativos
│   └── videos/                  # Grabaciones de evolución
└── tests/
    ├── test_neuron.py
    ├── test_automaton.py
    └── test_learning.py
```

---

## 📅 Plan de Desarrollo

### Fase 1: Autómata Celular Base (Notebook 1)

**Objetivo:** Implementar un autómata celular neuronal simple, más cercano a Conway pero con sinapsis aprendibles.

**Componentes:**

1. **Neurona básica:**
   - Estado: `valor` (0-1), `tension` (umbral)
   - Método: `procesar()` (agregar dendritas), `activar()` (umbral)

2. **Dendrita:**
   - Agregación de sinapsis (AND difuso = promedio)
   - Método: `procesar()`

3. **Sinapsis:**
   - Peso: `peso` (0-1)
   - Método: `procesar()` (similaridad pesada), `entrenar()` (Hebbian)

4. **Red (CNA):**
   - Grid 2D (ej: 64x64)
   - Conectividad local (vecindad 3x3 o 5x5)
   - Método: `step()` (un paso de tiempo)

5. **Visualización:**
   - Heatmap de activación neuronal
   - Animación de evolución temporal

**Experimentos:**

- Patrones oscilantes (como en Game of Life)
- Propagación de onda
- Reflejo simple (sensor → motor)

**Duración:** 1-2 semanas

---

### Fase 2: Mapas Auto-Organizados (Notebook 2)

**Objetivo:** Añadir Self-Organizing Maps (Kohonen) con inhibición lateral tipo sombrero mexicano.

**Componentes:**

1. **Mexican Hat Function:**
   ```python
   def mexican_hat(distance, sigma_excite=1.0, sigma_inhibit=3.0):
       excite = np.exp(-distance**2 / (2*sigma_excite**2))
       inhibit = 0.5 * np.exp(-distance**2 / (2*sigma_inhibit**2))
       return excite - inhibit
   ```

2. **Dendritas laterales:**
   - Excitación: Dendritas conectadas a vecinos cercanos (peso positivo)
   - Inhibición: Dendritas conectadas a vecinos lejanos (peso negativo)

3. **Aprendizaje competitivo:**
   - Winner-take-all (neurona con mayor activación)
   - Actualizar vecindad de ganador:
     ```python
     for neighbor in neighborhood(winner, radius=3):
         neighbor.synapses.weight += lr * (input - weight)
     ```

4. **Clustering:**
   - Entrenar con patrones de entrada (ej: dígitos MNIST)
   - Ver cómo emergen clusters topológicos

**Experimentos:**

- Mapeo de colores RGB → Grid 2D
- Clustering de embeddings de texto
- Mapas retinotópicos (imagen → activación espacial)

**Duración:** 1-2 semanas

---

### Fase 3: Memoria Temporal y Predicción (Notebook 3)

**Objetivo:** Implementar Hierarchical Temporal Memory (Hawkins) para aprender secuencias y predecir.

**Componentes:**

1. **Sequence Memory:**
   ```python
   class SequenceMemory:
       def __init__(self):
           self.sequences = {}  # (state_t, state_{t-1}) → count
       
       def learn(self, current, previous):
           key = (hash(current), hash(previous))
           self.sequences[key] = self.sequences.get(key, 0) + 1
       
       def predict(self, current):
           # Buscar qué sigue después de 'current'
           candidates = [
               (next_state, count) 
               for (next_state, prev_state), count in self.sequences.items()
               if prev_state == hash(current)
           ]
           return max(candidates, key=lambda x: x[1])[0] if candidates else None
   ```

2. **Predictive State:**
   - Neuronas en "estado predictivo" (tensión reducida)
   - Si la predicción acierta → refuerzo
   - Si falla → sorpresa → aprendizaje fuerte

3. **Columnas corticales:**
   - Grupos de neuronas que aprenden patrones
   - Jerarquía: Nivel 1 → Nivel 2 → Nivel 3

4. **Sparse Distributed Representation:**
   - Solo 2% de neuronas activas simultáneamente
   - Aumenta capacidad de memoria

**Experimentos:**

- Aprender secuencias ABC, DEF → predecir siguiente
- Navegar maze y predecir siguiente ubicación
- Memory replay: Reproducir secuencias sin input

**Duración:** 2-3 semanas

---

### Fase 4: UI Interactiva y Robótica (Notebook 4)

**Objetivo:** Crear interfaz para dibujar neuronas y controlar un robot simulado.

**Componentes:**

1. **Canvas interactivo (ipycanvas):**
   ```python
   from ipycanvas import Canvas
   canvas = Canvas(width=800, height=600)
   
   def on_mouse_down(x, y):
       grid_x, grid_y = canvas_to_grid(x, y)
       cna.grid[grid_x][grid_y].value = 1.0  # Activar neurona
       render()
   
   canvas.on_mouse_down(on_mouse_down)
   ```

2. **Controles (ipywidgets):**
   - Play/Pause/Step
   - Sliders: velocidad, learning rate, umbral
   - Dropdown: región (ENTRADA/SALIDA/INTERNA)
   - Brush size para dibujar

3. **Constructor de patrones:**
   - Cargar patrones predefinidos (como `conexionados.js`)
   - Guardar/cargar estados

4. **Robot simulado:**
   - Grid 32x32 con comida, obstáculos
   - Sensores: 8 direcciones (N, NE, E, SE, S, SW, W, NW)
   - Actuadores: Avanzar, girar
   - Objetivo: Aprender navegación con CNA

5. **Métricas:**
   - Plot de activación temporal
   - Histograma de pesos sinápticos
   - Heatmap de place cells

**Experimentos:**

- Dibujar patrones manualmente y ver evolución
- Entrenar robot a encontrar comida
- Comparar CNA vs. Policy Gradient (RL)

**Duración:** 2-3 semanas

---

### Fase 5 (Opcional): Integración con Transformers

**Objetivo:** Conectar CNA (bajo nivel) con Transformers (alto nivel).

**Arquitectura:**

```python
class HybridBrain:
    def __init__(self):
        self.cna = ConnessionistNeuralAutomaton(64, 64)
        self.transformer = AutoModel.from_pretrained("bert-base")
        self.embedding_bridge = nn.Linear(768, 64*64)  # BERT → CNA
        self.action_bridge = nn.Linear(64*64, 512)     # CNA → BERT
    
    def think(self, language_input, sensory_input):
        # 1. Procesar lenguaje (Transformer)
        lang_embedding = self.transformer(language_input).last_hidden_state.mean(1)
        
        # 2. Traducir a activación neuronal
        cna_pattern = self.embedding_bridge(lang_embedding).view(64, 64)
        self.cna.set_region('INTERNA', cna_pattern)
        
        # 3. Procesar sensores + lenguaje en CNA
        self.cna.set_region('ENTRADA', sensory_input)
        for _ in range(10):
            self.cna.step()
        
        # 4. Leer acción y traducir a lenguaje
        action_pattern = self.cna.get_region('SALIDA').flatten()
        action_embedding = self.action_bridge(action_pattern)
        
        return action_embedding  # Puede conectar a decoder para generar texto
```

**Ejemplo de uso:**

```python
brain = HybridBrain()

# Instrucción en lenguaje natural
instruction = tokenizer("Move to the red object", return_tensors="pt")

# Sensores del robot
sensors = torch.tensor([[0.2, 0.8, 0.0, 0.1, ...]])  # Valores normalizados

# Pensar
action_embedding = brain.think(instruction, sensors)

# Ejecutar
action = action_decoder(action_embedding)  # "move_forward", "turn_left", etc.
robot.execute(action)
```

**Duración:** 3-4 semanas

---

## 📘 Notebook 1: Autómata Celular Base

### Objetivos

1. Implementar clases básicas: `Neurona`, `Dendrita`, `Sinapsis`
2. Crear `CNA` (Connectionist Neural Automaton)
3. Visualizar evolución temporal
4. Experimentos con patrones oscilantes y propagación

### Instalación

```python
# Celda 1: Instalación
!pip install torch torchvision numpy matplotlib ipycanvas ipywidgets
```

### Configuración

```python
# Celda 2: Imports y configuración
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import ipywidgets as widgets
from ipycanvas import Canvas
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando: {device}")

# Reproducibilidad
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### Clases Base

```python
# Celda 3: Configuración global
@dataclass
class Config:
    """Configuración global del sistema (como cfg/config.js)"""
    
    # Coeficientes de aprendizaje
    COEF_SINAPSIS_ENTRENAMIENTO: float = 0.1
    COEF_SINAPSIS_UMBRAL_PESO: float = 0.05
    COEF_SINAPSIS_PESO_MEDIO: float = 0.5
    
    # Umbrales de activación
    NEURONA_UMBRAL_ACTIVACION: float = 0.5
    NEURONA_TENSION_INICIAL: float = 1.0
    
    # Coeficientes de tensión
    COEF_NEURONA_TENSION_RECUPERACION: float = 0.01
    COEF_NEURONA_TENSION_DISIPACION: float = 0.05
    
    # Regiones
    REGIONES: dict = None
    
    def __post_init__(self):
        if self.REGIONES is None:
            self.REGIONES = {
                'ENTRADA': 0,
                'SALIDA': 1,
                'INTERNA': 2,
                'DOLOR': 3
            }

config = Config()
```

```python
# Celda 4: Clase Sinapsis
class Sinapsis:
    """
    Conexión entre dos neuronas con peso Hebbiano.
    Inspirado en: model/Sinapsis.js del proyecto original
    """
    
    def __init__(self, neurona_origen, peso: float = None):
        self.origen = neurona_origen  # Neurona presináptica
        self.destino = None           # Se asigna cuando se conecta a dendrita
        
        # Peso inicial aleatorio o especificado
        if peso is None:
            self.peso = random.uniform(0.3, 0.7)
        else:
            self.peso = peso
        
        # Para visualización
        self.ultima_activacion = 0.0
    
    def procesar(self) -> float:
        """
        Calcula el valor que aporta esta sinapsis.
        Original: Usa similaridad entre origen y destino.
        """
        if self.origen is None:
            return 0.0
        
        valor_origen = self.origen.valor
        
        # Si hay destino, usar similaridad (como en original)
        if self.destino is not None:
            valor_destino = self.destino.valor
            similaridad = 1.0 - abs(valor_origen - valor_destino)
            self.ultima_activacion = self.peso * similaridad
        else:
            # Sin destino, simplemente pasar valor pesado
            self.ultima_activacion = self.peso * valor_origen
        
        return self.ultima_activacion
    
    def entrenar(self):
        """
        Aprendizaje Hebbiano: "Neurons that fire together, wire together"
        w += η * pre * post
        """
        if self.origen is None or self.destino is None:
            return
        
        pre = self.origen.valor
        post = self.destino.valor
        
        # Regla Hebbiana
        delta = config.COEF_SINAPSIS_ENTRENAMIENTO * pre * post
        self.peso += delta
        
        # Clamp peso entre [0, 1]
        self.peso = max(0.0, min(1.0, self.peso))
        
        # Poda sináptica (eliminar sinapsis débiles)
        if self.peso < config.COEF_SINAPSIS_UMBRAL_PESO:
            self.peso = 0.0
    
    def __repr__(self):
        return f"Sinapsis(peso={self.peso:.3f})"
```

```python
# Celda 5: Clase Dendrita
class Dendrita:
    """
    Agrupación de sinapsis que implementa lógica AND difusa.
    Inspirado en: model/Dendrita.js del proyecto original
    """
    
    def __init__(self, sinapsis: List[Sinapsis] = None):
        self.sinapsis = sinapsis if sinapsis is not None else []
        self.valor = 0.0
    
    def procesar(self) -> float:
        """
        Procesa todas las sinapsis y agrega sus valores.
        Original: Promedio de sinapsis activas (AND difuso)
        """
        if not self.sinapsis:
            self.valor = 0.0
            return self.valor
        
        # Filtrar sinapsis con peso > 0
        activas = [s for s in self.sinapsis if s.peso > 0]
        
        if not activas:
            self.valor = 0.0
            return self.valor
        
        # AND difuso: Promedio de valores de sinapsis
        suma = sum(s.procesar() for s in activas)
        self.valor = suma / len(activas)
        
        return self.valor
    
    def entrenar(self):
        """Entrena todas las sinapsis de esta dendrita"""
        for sinapsis in self.sinapsis:
            sinapsis.entrenar()
    
    def agregar_sinapsis(self, sinapsis: Sinapsis):
        """Añade una sinapsis a esta dendrita"""
        self.sinapsis.append(sinapsis)
        sinapsis.destino = self  # Asignar referencia inversa
    
    def __repr__(self):
        return f"Dendrita(valor={self.valor:.3f}, sinapsis={len(self.sinapsis)})"
```

```python
# Celda 6: Clase Neurona
class Neurona:
    """
    Unidad básica de procesamiento del CNA.
    Inspirado en: model/Neurona.js del proyecto original
    """
    
    def __init__(self, x: int, y: int, region: int = 2):
        # Posición en el grid
        self.x = x
        self.y = y
        
        # Región funcional
        self.region = region  # 0=ENTRADA, 1=SALIDA, 2=INTERNA, 3=DOLOR
        
        # Estado
        self.valor = 0.0      # Activación actual
        self.activa = False    # Si está activada (valor > umbral)
        
        # Tensión superficial (umbral dinámico)
        self.tension = config.NEURONA_TENSION_INICIAL
        
        # Dendritas (entradas)
        self.dendritas: List[Dendrita] = []
        
        # Historial (para visualización)
        self.historial_valor = []
    
    def procesar(self):
        """
        Procesa todas las dendritas y calcula el nuevo valor.
        Original: OR difuso = máximo de dendritas
        """
        if not self.dendritas:
            return
        
        # Procesar cada dendrita
        valores_dendritas = [d.procesar() for d in self.dendritas]
        
        # OR difuso: Máximo
        if valores_dendritas:
            self.valor = max(valores_dendritas)
        else:
            self.valor = 0.0
    
    def activar(self):
        """
        Compara valor con tensión (umbral) y decide si activar.
        """
        if self.valor >= self.tension:
            self.activa = True
            self.valor = 1.0  # Activación completa
            
            # Disipar tensión (periodo refractario)
            self.tension = max(0.1, self.tension - config.COEF_NEURONA_TENSION_DISIPACION)
        else:
            self.activa = False
            
            # Recuperar tensión gradualmente
            if self.tension < config.NEURONA_TENSION_INICIAL:
                self.tension += config.COEF_NEURONA_TENSION_RECUPERACION
    
    def entrenar(self):
        """Entrena todas las dendritas de esta neurona"""
        for dendrita in self.dendritas:
            dendrita.entrenar()
    
    def reset(self):
        """Resetea el estado de la neurona"""
        self.valor = 0.0
        self.activa = False
        self.tension = config.NEURONA_TENSION_INICIAL
    
    def __repr__(self):
        return f"Neurona({self.x},{self.y}, v={self.valor:.2f}, t={self.tension:.2f})"


class NeuronaEntrada(Neurona):
    """
    Neurona de entrada (sensorial).
    No procesa dendritas, su valor se setea externamente.
    """
    
    def __init__(self, x: int, y: int):
        super().__init__(x, y, region=0)  # Región ENTRADA
    
    def procesar(self):
        # No procesar dendritas, el valor viene del exterior
        pass
    
    def set_valor(self, valor: float):
        """Establece el valor directamente (desde sensores)"""
        self.valor = max(0.0, min(1.0, valor))
        self.activar()
```

### Autómata Celular

```python
# Celda 7: Clase CNA
class ConnessionistNeuralAutomaton:
    """
    Autómata Celular Neuronal principal.
    Grid 2D de neuronas con conectividad local.
    """
    
    def __init__(self, width: int = 64, height: int = 64, connect_radius: int = 3):
        self.width = width
        self.height = height
        self.connect_radius = connect_radius
        
        # Crear grid de neuronas
        self.grid = [[Neurona(x, y) for x in range(width)] for y in range(height)]
        
        # Aplanar para iteración fácil
        self.neuronas = [n for row in self.grid for n in row]
        
        # Definir regiones
        self.definir_regiones()
        
        # Conectar neuronas
        self.conectar_localmente(radius=connect_radius)
        
        # Estadísticas
        self.paso_actual = 0
        self.learning_enabled = True
    
    def definir_regiones(self):
        """
        Define regiones funcionales en el grid.
        Similar a setupRegiones.js
        """
        h = self.height
        
        # ENTRADA: Arriba (primeras 16 filas)
        for y in range(min(16, h//4)):
            for x in range(self.width):
                self.grid[y][x].region = config.REGIONES['ENTRADA']
                self.grid[y][x] = NeuronaEntrada(x, y)
        
        # SALIDA: Abajo (últimas 16 filas)
        for y in range(max(0, h - 16), h):
            for x in range(self.width):
                self.grid[y][x].region = config.REGIONES['SALIDA']
        
        # DOLOR: Esquina inferior derecha
        for y in range(max(0, h - 8), h):
            for x in range(max(0, self.width - 8), self.width):
                self.grid[y][x].region = config.REGIONES['DOLOR']
        
        # INTERNA: Todo lo demás (ya tiene región 2 por defecto)
    
    def get_neighbors(self, neuron: Neurona, radius: int = 3) -> List[Neurona]:
        """Obtiene vecinos dentro de un radio"""
        neighbors = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = neuron.x + dx, neuron.y + dy
                
                # Toroidal wrap (opcional)
                # nx = nx % self.width
                # ny = ny % self.height
                
                # Clamp (no wrap)
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append(self.grid[ny][nx])
        
        return neighbors
    
    def conectar_localmente(self, radius: int = 3):
        """
        Conecta cada neurona con sus vecinos cercanos.
        Cada neurona tiene varias dendritas, cada dendrita tiene varias sinapsis.
        """
        num_dendritas = 4  # Como en el original
        
        for neuron in self.neuronas:
            # Saltar neuronas de entrada (no tienen dendritas)
            if isinstance(neuron, NeuronaEntrada):
                continue
            
            neighbors = self.get_neighbors(neuron, radius)
            
            if not neighbors:
                continue
            
            # Crear dendritas
            for _ in range(num_dendritas):
                dendrita = Dendrita()
                
                # Cada dendrita se conecta a subset de vecinos
                num_sinapsis = random.randint(2, min(8, len(neighbors)))
                vecinos_seleccionados = random.sample(neighbors, num_sinapsis)
                
                for vecino in vecinos_seleccionados:
                    sinapsis = Sinapsis(vecino)
                    sinapsis.destino = neuron
                    dendrita.agregar_sinapsis(sinapsis)
                
                neuron.dendritas.append(dendrita)
    
    def step(self):
        """
        Un paso de tiempo del autómata.
        Similar a red.procesar() en model/Red.js
        """
        # 1. Procesar todas las neuronas (excepto ENTRADA)
        for neuron in self.neuronas:
            if not isinstance(neuron, NeuronaEntrada):
                neuron.procesar()
        
        # 2. Activar neuronas
        for neuron in self.neuronas:
            neuron.activar()
        
        # 3. Entrenar sinapsis (si aprendizaje habilitado)
        if self.learning_enabled:
            for neuron in self.neuronas:
                neuron.entrenar()
        
        self.paso_actual += 1
    
    def set_input_region(self, values: np.ndarray):
        """
        Establece valores en la región de ENTRADA.
        values: Array 2D (H x W) o 1D que se mapea al ancho
        """
        entrada_neurons = [n for n in self.neuronas if n.region == config.REGIONES['ENTRADA']]
        
        if values.ndim == 1:
            # Vector 1D: Mapear a ancho de región ENTRADA
            for i, neuron in enumerate(entrada_neurons):
                if i < len(values):
                    neuron.set_valor(values[i])
        else:
            # Array 2D: Mapear directamente
            for neuron in entrada_neurons:
                if neuron.y < values.shape[0] and neuron.x < values.shape[1]:
                    neuron.set_valor(values[neuron.y, neuron.x])
    
    def get_output_region(self) -> np.ndarray:
        """Obtiene valores de la región de SALIDA"""
        salida_neurons = [n for n in self.neuronas if n.region == config.REGIONES['SALIDA']]
        salida_neurons.sort(key=lambda n: (n.y, n.x))
        return np.array([n.valor for n in salida_neurons])
    
    def get_state(self) -> np.ndarray:
        """Obtiene el estado completo del grid como array 2D"""
        state = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                state[y, x] = self.grid[y][x].valor
        return state
    
    def reset(self):
        """Resetea todas las neuronas"""
        for neuron in self.neuronas:
            neuron.reset()
        self.paso_actual = 0
```

### Visualización

```python
# Celda 8: Visualización con Matplotlib
def visualizar_cna(cna: ConnessionistNeuralAutomaton, figsize=(12, 10)):
    """
    Visualiza el estado del CNA con regiones coloreadas.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Activación neuronal
    state = cna.get_state()
    im1 = axes[0, 0].imshow(state, cmap='hot', vmin=0, vmax=1)
    axes[0, 0].set_title(f'Activación Neuronal (t={cna.paso_actual})')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Regiones
    regions = np.zeros((cna.height, cna.width))
    for y in range(cna.height):
        for x in range(cna.width):
            regions[y, x] = cna.grid[y][x].region
    
    im2 = axes[0, 1].imshow(regions, cmap='tab10', vmin=0, vmax=3)
    axes[0, 1].set_title('Regiones (ENTRADA=0, SALIDA=1, INTERNA=2, DOLOR=3)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], ticks=[0, 1, 2, 3])
    
    # 3. Tensión neuronal
    tension = np.zeros((cna.height, cna.width))
    for y in range(cna.height):
        for x in range(cna.width):
            tension[y, x] = cna.grid[y][x].tension
    
    im3 = axes[1, 0].imshow(tension, cmap='viridis', vmin=0, vmax=1)
    axes[1, 0].set_title('Tensión (Umbral dinámico)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 4. Histograma de pesos sinápticos
    pesos = []
    for neuron in cna.neuronas:
        for dendrita in neuron.dendritas:
            pesos.extend([s.peso for s in dendrita.sinapsis])
    
    axes[1, 1].hist(pesos, bins=50, color='steelblue', alpha=0.7)
    axes[1, 1].set_title(f'Distribución de Pesos Sinápticos (N={len(pesos)})')
    axes[1, 1].set_xlabel('Peso')
    axes[1, 1].set_ylabel('Frecuencia')
    
    plt.tight_layout()
    return fig

# Prueba inicial
print("Creando CNA de prueba 32x32...")
cna_test = ConnessionistNeuralAutomaton(32, 32, connect_radius=2)
print(f"Creado con {len(cna_test.neuronas)} neuronas")

# Contar sinapsis
total_sinapsis = sum(
    len(d.sinapsis) 
    for n in cna_test.neuronas 
    for d in n.dendritas
)
print(f"Total de sinapsis: {total_sinapsis}")

# Visualizar estado inicial
visualizar_cna(cna_test)
plt.show()
```

### Experimentos

```python
# Celda 9: Experimento 1 - Propagación de onda
def experimento_propagacion():
    """
    Activa una neurona central y observa cómo se propaga la activación.
    """
    print("=== Experimento 1: Propagación de Onda ===\n")
    
    cna = ConnessionistNeuralAutomaton(48, 48, connect_radius=3)
    
    # Activar neurona central
    center_x, center_y = cna.width // 2, cna.height // 2
    cna.grid[center_y][center_x].valor = 1.0
    cna.grid[center_y][center_x].activa = True
    
    # Deshabilitar aprendizaje para ver propagación pura
    cna.learning_enabled = False
    
    # Simular 20 pasos
    num_steps = 20
    states = [cna.get_state().copy()]
    
    for i in range(num_steps):
        cna.step()
        states.append(cna.get_state().copy())
    
    # Visualizar evolución
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, state in enumerate(states):
        im = axes[i].imshow(state, cmap='hot', vmin=0, vmax=1)
        axes[i].set_title(f't={i}')
        axes[i].axis('off')
    
    plt.suptitle('Propagación de Onda desde Centro', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return cna

experimento_propagacion()
```

```python
# Celda 10: Experimento 2 - Reflejo simple (sensor → motor)
def experimento_reflejo():
    """
    Entrena un reflejo simple: sensor activo → motor activo.
    Similar al reflejo de retracción de Aplysia.
    """
    print("=== Experimento 2: Reflejo Simple ===\n")
    
    cna = ConnessionistNeuralAutomaton(32, 32, connect_radius=3)
    cna.learning_enabled = True
    
    # Entrenar: Activar sensores repetidamente
    num_epochs = 100
    activaciones_salida = []
    
    for epoch in range(num_epochs):
        # Reset
        cna.reset()
        
        # Activar toda la región ENTRADA
        input_pattern = np.ones(cna.width) * 0.8
        cna.set_input_region(input_pattern)
        
        # Propagar durante 10 pasos
        for _ in range(10):
            cna.step()
        
        # Medir activación en región SALIDA
        output = cna.get_output_region()
        activaciones_salida.append(output.mean())
    
    # Plot aprendizaje
    plt.figure(figsize=(10, 5))
    plt.plot(activaciones_salida, linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Activación media de región SALIDA')
    plt.title('Aprendizaje de Reflejo: Sensor → Motor')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Activación inicial: {activaciones_salida[0]:.4f}")
    print(f"Activación final: {activaciones_salida[-1]:.4f}")
    print(f"Incremento: {activaciones_salida[-1] - activaciones_salida[0]:.4f}")
    
    # Visualizar estado final
    visualizar_cna(cna)
    plt.show()
    
    return cna

experimento_reflejo()
```

```python
# Celda 11: Experimento 3 - Patrón oscilante
def experimento_oscilador():
    """
    Busca patrones oscilantes emergentes (como en Game of Life).
    """
    print("=== Experimento 3: Patrones Oscilantes ===\n")
    
    cna = ConnessionistNeuralAutomaton(32, 32, connect_radius=2)
    cna.learning_enabled = False
    
    # Crear patrón inicial: Blinker vertical
    #   ●
    #   ●
    #   ●
    center_x, center_y = cna.width // 2, cna.height // 2
    for dy in [-1, 0, 1]:
        cna.grid[center_y + dy][center_x].valor = 1.0
    
    # Simular
    num_steps = 10
    states = []
    
    for i in range(num_steps):
        states.append(cna.get_state().copy())
        cna.step()
    
    # Detectar periodo
    def compare_states(s1, s2):
        return np.allclose(s1, s2, atol=0.1)
    
    periodo = None
    for i in range(1, len(states)):
        if compare_states(states[0], states[i]):
            periodo = i
            break
    
    # Visualizar
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, state in enumerate(states):
        im = axes[i].imshow(state, cmap='hot', vmin=0, vmax=1)
        axes[i].set_title(f't={i}')
        axes[i].axis('off')
    
    if periodo:
        plt.suptitle(f'Oscilador con Periodo {periodo}', fontsize=16)
    else:
        plt.suptitle('Evolución del Patrón', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    return cna

experimento_oscilador()
```

### Análisis y Métricas

```python
# Celda 12: Análisis de conectividad
def analizar_conectividad(cna: ConnessionistNeuralAutomaton):
    """Analiza la estructura de conectividad del CNA"""
    
    print("=== Análisis de Conectividad ===\n")
    
    # Contar dendritas por neurona
    dendritas_por_neurona = [len(n.dendritas) for n in cna.neuronas if not isinstance(n, NeuronaEntrada)]
    
    # Contar sinapsis por dendrita
    sinapsis_por_dendrita = []
    for n in cna.neuronas:
        for d in n.dendritas:
            sinapsis_por_dendrita.append(len(d.sinapsis))
    
    # Distribución de pesos
    pesos = []
    for n in cna.neuronas:
        for d in n.dendritas:
            pesos.extend([s.peso for s in d.sinapsis])
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Dendritas por neurona
    axes[0].hist(dendritas_por_neurona, bins=10, color='steelblue', alpha=0.7)
    axes[0].set_title('Dendritas por Neurona')
    axes[0].set_xlabel('Número de dendritas')
    axes[0].set_ylabel('Frecuencia')
    axes[0].axvline(np.mean(dendritas_por_neurona), color='red', linestyle='--', 
                    label=f'Media: {np.mean(dendritas_por_neurona):.1f}')
    axes[0].legend()
    
    # 2. Sinapsis por dendrita
    axes[1].hist(sinapsis_por_dendrita, bins=15, color='seagreen', alpha=0.7)
    axes[1].set_title('Sinapsis por Dendrita')
    axes[1].set_xlabel('Número de sinapsis')
    axes[1].set_ylabel('Frecuencia')
    axes[1].axvline(np.mean(sinapsis_por_dendrita), color='red', linestyle='--',
                    label=f'Media: {np.mean(sinapsis_por_dendrita):.1f}')
    axes[1].legend()
    
    # 3. Distribución de pesos
    axes[2].hist(pesos, bins=50, color='coral', alpha=0.7)
    axes[2].set_title('Distribución de Pesos Sinápticos')
    axes[2].set_xlabel('Peso')
    axes[2].set_ylabel('Frecuencia')
    axes[2].axvline(np.mean(pesos), color='red', linestyle='--',
                    label=f'Media: {np.mean(pesos):.3f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas
    print(f"Total de neuronas: {len(cna.neuronas)}")
    print(f"Total de dendritas: {sum(dendritas_por_neurona)}")
    print(f"Total de sinapsis: {len(pesos)}")
    print(f"\nDendritas por neurona: {np.mean(dendritas_por_neurona):.2f} ± {np.std(dendritas_por_neurona):.2f}")
    print(f"Sinapsis por dendrita: {np.mean(sinapsis_por_dendrita):.2f} ± {np.std(sinapsis_por_dendrita):.2f}")
    print(f"Peso sináptico medio: {np.mean(pesos):.4f} ± {np.std(pesos):.4f}")
    print(f"\nSparsity: {(np.array(pesos) == 0).mean() * 100:.1f}% de sinapsis eliminadas (peso=0)")

# Analizar CNA de prueba
analizar_conectividad(cna_test)
```

### Guardar y Cargar

```python
# Celda 13: Guardar/cargar estado
import pickle

def guardar_cna(cna: ConnessionistNeuralAutomaton, filepath: str):
    """Guarda el estado completo del CNA"""
    with open(filepath, 'wb') as f:
        pickle.dump(cna, f)
    print(f"CNA guardado en: {filepath}")

def cargar_cna(filepath: str) -> ConnessionistNeuralAutomaton:
    """Carga un CNA guardado"""
    with open(filepath, 'rb') as f:
        cna = pickle.load(f)
    print(f"CNA cargado de: {filepath}")
    return cna

# Ejemplo
# guardar_cna(cna_test, "cna_estado.pkl")
# cna_cargado = cargar_cna("cna_estado.pkl")
```

### Resumen del Notebook 1

```markdown
## ✅ Logros del Notebook 1

1. **Clases base implementadas:**
   - `Sinapsis`: Conexión pesada con aprendizaje Hebbiano
   - `Dendrita`: Agrupación de sinapsis (AND difuso)
   - `Neurona`: Unidad de procesamiento con tensión dinámica
   - `NeuronaEntrada`: Neurona sensorial

2. **Autómata celular:**
   - `ConnessionistNeuralAutomaton`: Grid 2D con regiones funcionales
   - Conectividad local (vecindad configurable)
   - Método `step()` para evolución temporal

3. **Experimentos:**
   - Propagación de onda desde centro
   - Reflejo simple (aprendizaje sensor→motor)
   - Patrones oscilantes

4. **Visualización:**
   - Heatmaps de activación, tensión, regiones
   - Histogramas de pesos sinápticos
   - Análisis de conectividad

## 🚀 Próximos Pasos (Notebook 2)

- Implementar **Self-Organizing Maps** (Kohonen)
- Añadir **inhibición lateral** (sombrero mexicano)
- Experimentos de **clustering** espontáneo
- Visualizar mapas topológicos emergentes
```

---

## 📘 Notebook 2: Mapas Auto-Organizados (Kohonen)

### Objetivos

1. Implementar **Self-Organizing Map** (Kohonen)
2. Función **Mexican Hat** (sombrero mexicano) para inhibición lateral
3. **Clustering espontáneo** de patrones
4. Visualizar **mapas topológicos** emergentes

### Concepto: ¿Qué es un SOM?

```
ENTRADA: Patrones de alta dimensión
         ↓
    [Neurona 1][Neurona 2][Neurona 3]
    [Neurona 4][Neurona 5][Neurona 6]  ← Grid 2D
    [Neurona 7][Neurona 8][Neurona 9]
         ↓
SALIDA: Representación 2D donde patrones similares
        activan neuronas VECINAS (topología preservada)
```

**Ejemplo biológico:** Corteza auditiva tonotópica
- Frecuencias similares → neuronas vecinas
- Mapa ordenado emerge del aprendizaje

### Implementación

```python
# Celda 1: Función Mexican Hat
import numpy as np
import matplotlib.pyplot as plt

def mexican_hat(distance: float, sigma_excite: float = 1.0, sigma_inhibit: float = 3.0, 
                amplitude_excite: float = 1.0, amplitude_inhibit: float = 0.5) -> float:
    """
    Función de activación tipo sombrero mexicano.
    
    Args:
        distance: Distancia euclidiana desde el centro
        sigma_excite: Amplitud de la excitación central
        sigma_inhibit: Amplitud de la inhibición lateral
        amplitude_excite: Altura del pico de excitación
        amplitude_inhibit: Profundidad de la inhibición
    
    Returns:
        Valor de activación (positivo = excitación, negativo = inhibición)
    """
    # Componente de excitación (Gaussiana estrecha)
    excitation = amplitude_excite * np.exp(-distance**2 / (2 * sigma_excite**2))
    
    # Componente de inhibición (Gaussiana ancha)
    inhibition = amplitude_inhibit * np.exp(-distance**2 / (2 * sigma_inhibit**2))
    
    return excitation - inhibition

# Visualizar la función
distances = np.linspace(0, 10, 200)
activations = [mexican_hat(d) for d in distances]

plt.figure(figsize=(10, 6))
plt.plot(distances, activations, linewidth=3, color='darkblue')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.fill_between(distances, 0, activations, where=np.array(activations) > 0, 
                 color='green', alpha=0.3, label='Excitación')
plt.fill_between(distances, 0, activations, where=np.array(activations) < 0, 
                 color='red', alpha=0.3, label='Inhibición')
plt.xlabel('Distancia desde neurona ganadora', fontsize=12)
plt.ylabel('Efecto en neuronas vecinas', fontsize=12)
plt.title('Función Mexican Hat (Sombrero Mexicano)', fontsize=14, weight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()
```

```python
# Celda 2: Clase KohonenSOM
class KohonenSOM:
    """
    Self-Organizing Map de Kohonen.
    Implementa clustering topológico con Mexican Hat.
    """
    
    def __init__(self, map_size: Tuple[int, int], input_dim: int):
        """
        Args:
            map_size: (height, width) del mapa 2D
            input_dim: Dimensionalidad del input
        """
        self.height, self.width = map_size
        self.input_dim = input_dim
        
        # Pesos: Cada neurona tiene un vector de pesos de tamaño input_dim
        # Inicialización aleatoria pequeña
        self.weights = np.random.randn(self.height, self.width, input_dim) * 0.1
        
        # Para visualización
        self.activations = np.zeros((self.height, self.width))
        self.winner_history = []
    
    def find_winner(self, input_vector: np.ndarray) -> Tuple[int, int]:
        """
        Encuentra la neurona ganadora (BMU - Best Matching Unit).
        La neurona cuyos pesos están más cerca del input.
        """
        # Calcular distancia euclidiana de cada neurona al input
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        
        # Neurona con menor distancia
        winner_idx = np.unravel_index(np.argmin(distances), distances.shape)
        
        return winner_idx
    
    def get_neighborhood(self, winner: Tuple[int, int], radius: float) -> np.ndarray:
        """
        Calcula la función de vecindad (Mexican Hat).
        
        Returns:
            Array (H x W) con valores de influencia para cada neurona
        """
        wy, wx = winner
        
        # Crear grid de distancias
        y_grid, x_grid = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
        distances = np.sqrt((y_grid - wy)**2 + (x_grid - wx)**2)
        
        # Aplicar Mexican Hat
        neighborhood = np.vectorize(mexican_hat)(
            distances, 
            sigma_excite=radius, 
            sigma_inhibit=radius*2
        )
        
        return neighborhood
    
    def update(self, input_vector: np.ndarray, learning_rate: float, radius: float):
        """
        Actualiza los pesos según el algoritmo de Kohonen.
        """
        # 1. Encontrar ganador
        winner = self.find_winner(input_vector)
        self.winner_history.append(winner)
        
        # 2. Calcular función de vecindad
        neighborhood = self.get_neighborhood(winner, radius)
        
        # 3. Actualizar pesos
        # w_new = w_old + lr * neighborhood * (input - w_old)
        for i in range(self.height):
            for j in range(self.width):
                influence = neighborhood[i, j]
                if influence > 0:  # Solo actualizar si hay excitación
                    self.weights[i, j] += learning_rate * influence * (input_vector - self.weights[i, j])
    
    def train(self, data: np.ndarray, num_epochs: int, 
              initial_lr: float = 0.5, initial_radius: float = 3.0):
        """
        Entrena el SOM con un dataset.
        
        Args:
            data: Array (N x input_dim) con N ejemplos
            num_epochs: Número de épocas
            initial_lr: Learning rate inicial (decae con el tiempo)
            initial_radius: Radio de vecindad inicial (decae con el tiempo)
        """
        num_samples = len(data)
        
        for epoch in range(num_epochs):
            # Decaimiento exponencial
            lr = initial_lr * np.exp(-epoch / num_epochs)
            radius = initial_radius * np.exp(-epoch / num_epochs)
            
            # Presentar todos los ejemplos en orden aleatorio
            indices = np.random.permutation(num_samples)
            
            for idx in indices:
                input_vector = data[idx]
                self.update(input_vector, lr, radius)
            
            if (epoch + 1) % 10 == 0:
                print(f"Época {epoch+1}/{num_epochs} - LR: {lr:.4f}, Radius: {radius:.2f}")
    
    def get_activation_map(self, input_vector: np.ndarray) -> np.ndarray:
        """Calcula un mapa de activación para un input específico"""
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        # Convertir distancias a activaciones (más cerca = más activo)
        activations = np.exp(-distances / 2)
        return activations
    
    def visualize_weights(self, feature_names=None):
        """Visualiza los pesos de cada neurona"""
        fig, axes = plt.subplots(1, min(self.input_dim, 4), figsize=(15, 4))
        
        if self.input_dim == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            if i >= self.input_dim:
                break
            
            im = ax.imshow(self.weights[:, :, i], cmap='coolwarm')
            title = feature_names[i] if feature_names else f'Feature {i}'
            ax.set_title(title)
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.show()
```

```python
# Celda 3: Experimento 1 - Clustering de colores RGB
def experimento_som_colores():
    """
    Entrenar SOM para organizar colores RGB en un mapa 2D.
    Resultado: Colores similares aparecerán en regiones cercanas.
    """
    print("=== Experimento SOM: Clustering de Colores RGB ===\n")
    
    # Generar colores aleatorios (RGB normalizado)
    np.random.seed(42)
    num_colors = 500
    colors = np.random.rand(num_colors, 3)  # [R, G, B] en [0, 1]
    
    # Crear y entrenar SOM
    som = KohonenSOM(map_size=(20, 20), input_dim=3)
    som.train(colors, num_epochs=100, initial_lr=0.5, initial_radius=5.0)
    
    # Visualizar mapa de colores aprendido
    plt.figure(figsize=(12, 10))
    
    # Crear imagen RGB desde los pesos
    color_map = som.weights.copy()
    color_map = np.clip(color_map, 0, 1)  # Asegurar rango [0,1]
    
    plt.imshow(color_map)
    plt.title('Mapa Auto-Organizado de Colores RGB', fontsize=14, weight='bold')
    plt.axis('off')
    
    # Añadir grid para ver células
    for i in range(som.height + 1):
        plt.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
    for j in range(som.width + 1):
        plt.axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Visualizar activación para colores específicos
    test_colors = [
        ([1, 0, 0], "Rojo"),
        ([0, 1, 0], "Verde"),
        ([0, 0, 1], "Azul"),
        ([1, 1, 0], "Amarillo")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, (color, name) in enumerate(test_colors):
        activation = som.get_activation_map(np.array(color))
        
        im = axes[idx].imshow(activation, cmap='hot')
        axes[idx].set_title(f'Activación para {name} {color}', fontsize=12)
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    plt.show()
    
    return som

som_colores = experimento_som_colores()
```

```python
# Celda 4: Integrar SOM con CNA
class CNA_ConSOM(ConnessionistNeuralAutomaton):
    """
    CNA con capacidades de Self-Organization (Kohonen).
    Añade inhibición lateral tipo Mexican Hat.
    """
    
    def __init__(self, width: int = 64, height: int = 64):
        super().__init__(width, height, connect_radius=3)
        
        # Añadir dendritas laterales (para inhibición/excitación)
        self.add_lateral_connections()
    
    def add_lateral_connections(self):
        """
        Añade conexiones laterales a cada neurona según Mexican Hat.
        """
        for neuron in self.neuronas:
            if isinstance(neuron, NeuronaEntrada):
                continue
            
            # Crear dendrita lateral
            dendrita_lateral = Dendrita()
            
            # Conectar con vecinos en radio amplio
            neighbors = self.get_neighbors(neuron, radius=5)
            
            for neighbor in neighbors:
                # Calcular distancia
                dx = neighbor.x - neuron.x
                dy = neighbor.y - neuron.y
                distance = np.sqrt(dx**2 + dy**2)
                
                # Peso según Mexican Hat
                weight = mexican_hat(distance, sigma_excite=1.5, sigma_inhibit=3.0)
                
                # Crear sinapsis (puede ser negativa para inhibición)
                sinapsis = Sinapsis(neighbor, peso=abs(weight))
                sinapsis.inhibitoria = weight < 0  # Marcar como inhibitoria
                
                dendrita_lateral.agregar_sinapsis(sinapsis)
            
            neuron.dendritas.append(dendrita_lateral)
    
    def step_with_lateral(self):
        """Step con inhibición lateral explícita"""
        # Paso normal
        self.step()
        
        # Aplicar inhibición lateral adicional
        for neuron in self.neuronas:
            if isinstance(neuron, NeuronaEntrada):
                continue
            
            # Calcular efecto lateral
            lateral_effect = 0.0
            for dendrita in neuron.dendritas:
                for sinapsis in dendrita.sinapsis:
                    effect = sinapsis.procesar()
                    if hasattr(sinapsis, 'inhibitoria') and sinapsis.inhibitoria:
                        lateral_effect -= effect
                    else:
                        lateral_effect += effect
            
            # Modular valor
            neuron.valor = max(0.0, min(1.0, neuron.valor + lateral_effect * 0.1))

# Añadir atributo inhibitoria a Sinapsis existente
Sinapsis.inhibitoria = False
```

```python
# Celda 5: Experimento 2 - Clustering espontáneo en CNA
def experimento_clustering_espacial():
    """
    Mostrar cómo emergen clusters en el CNA con inhibición lateral.
    """
    print("=== Experimento: Clustering Espacial con Mexican Hat ===\n")
    
    cna = CNA_ConSOM(32, 32)
    cna.learning_enabled = False
    
    # Activar puntos aleatorios en región ENTRADA
    num_puntos = 50
    for _ in range(num_puntos):
        x = random.randint(0, cna.width - 1)
        y = random.randint(0, min(15, cna.height // 4))  # Región ENTRADA
        cna.grid[y][x].valor = random.uniform(0.5, 1.0)
    
    # Simular con inhibición lateral
    num_steps = 30
    states = []
    
    for i in range(num_steps):
        states.append(cna.get_state().copy())
        cna.step_with_lateral()
    
    # Visualizar evolución (muestrear cada 5 pasos)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    sample_steps = [0, 5, 10, 15, 20, 29]
    
    for idx, step in enumerate(sample_steps):
        im = axes[idx].imshow(states[step], cmap='hot', vmin=0, vmax=1)
        axes[idx].set_title(f't={step}', fontsize=12)
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx])
    
    plt.suptitle('Clustering Espontáneo con Inhibición Lateral', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()
    
    print("Observa cómo:")
    print("1. Activaciones iniciales dispersas")
    print("2. Inhibición lateral suprime vecinos")
    print("3. Emergen 'winner-take-all' clusters")
    print("4. Solo las neuronas más fuertes sobreviven")
    
    return cna

experimento_clustering_espacial()
```

```python
# Celda 6: Experimento 3 - Mapeo tonotópico (frecuencias)
def experimento_mapa_tonotopico():
    """
    Simula cómo la corteza auditiva crea mapas de frecuencias.
    Frecuencias similares → neuronas vecinas.
    """
    print("=== Experimento: Mapa Tonotópico (Frecuencias) ===\n")
    
    # Generar tonos de diferentes frecuencias
    num_frequencies = 50
    frequencies = np.linspace(100, 1000, num_frequencies)  # 100 Hz a 1000 Hz
    
    # Crear representaciones de frecuencia (one-hot like, pero suavizado)
    def frequency_to_vector(freq, all_freqs):
        # Gaussiana centrada en la frecuencia
        sigma = 50
        vector = np.exp(-((all_freqs - freq)**2) / (2 * sigma**2))
        return vector / vector.sum()
    
    freq_vectors = np.array([frequency_to_vector(f, frequencies) for f in frequencies])
    
    # Entrenar SOM
    som = KohonenSOM(map_size=(10, 50), input_dim=num_frequencies)
    som.train(freq_vectors, num_epochs=100, initial_lr=0.3, initial_radius=8.0)
    
    # Visualizar mapa tonotópico
    # Para cada posición del SOM, encontrar qué frecuencia responde mejor
    freq_map = np.zeros((som.height, som.width))
    
    for i in range(som.height):
        for j in range(som.width):
            # Pesos de esta neurona
            weights = som.weights[i, j]
            # Frecuencia con mayor peso
            best_freq_idx = np.argmax(weights)
            freq_map[i, j] = frequencies[best_freq_idx]
    
    # Plot
    plt.figure(figsize=(16, 6))
    
    im = plt.imshow(freq_map, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Frecuencia (Hz)')
    plt.title('Mapa Tonotópico Auto-Organizado\n(Similar a la corteza auditiva primaria)', 
              fontsize=14, weight='bold')
    plt.xlabel('Posición X en el mapa')
    plt.ylabel('Posición Y en el mapa')
    
    # Overlay: Marcar algunas frecuencias específicas
    test_freqs = [200, 400, 600, 800]
    for freq in test_freqs:
        activation = som.get_activation_map(frequency_to_vector(freq, frequencies))
        max_pos = np.unravel_index(np.argmax(activation), activation.shape)
        plt.plot(max_pos[1], max_pos[0], 'ro', markersize=10, 
                markeredgecolor='white', markeredgewidth=2)
        plt.text(max_pos[1], max_pos[0] - 1, f'{freq}Hz', 
                color='white', fontsize=10, ha='center', weight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("Observa:")
    print("- Frecuencias bajas (azul) → un extremo del mapa")
    print("- Frecuencias altas (amarillo) → otro extremo")
    print("- Topología preservada: Frecuencias vecinas → Neuronas vecinas")
    print("- ¡Como en el cerebro real!")
    
    return som

experimento_mapa_tonotopico()
```

### Resumen del Notebook 2

```markdown
## ✅ Logros del Notebook 2

1. **Self-Organizing Map (Kohonen):**
   - Implementación completa de SOM
   - Winner-take-all competitivo
   - Decaimiento de learning rate y radio

2. **Mexican Hat Function:**
   - Inhibición lateral + excitación central
   - Emergencia de clusters espaciales

3. **Integración con CNA:**
   - `CNA_ConSOM`: Autómata con conexiones laterales
   - Inhibición/excitación basada en distancia

4. **Experimentos:**
   - Clustering de colores RGB
   - Clustering espacial en grid 2D
   - Mapa tonotópico (frecuencias)

5. **Aprendizajes biológicos:**
   - Mapas topológicos emergen espontáneamente
   - Competencia neuronal → especialización
   - Preservación de topología del input

## 🚀 Próximos Pasos (Notebook 3)

- Implementar **Hierarchical Temporal Memory** (Hawkins)
- Aprendizaje de **secuencias temporales**
- **Predicción** del siguiente estado
- **Memory replay** sin input externo
```

---

## 📘 Notebook 3: Memoria Temporal y Predicción (HTM)

### Objetivos

1. Implementar **Temporal Memory** (secuencias)
2. **Predicción** del siguiente estado
3. **Sparse Distributed Representations** (SDR)
4. **Memory replay** (reproducir secuencias aprendidas)
5. Integrar con **place cells** y **grid cells**

### Concepto: Hierarchical Temporal Memory

Jeff Hawkins propone que el neocórtex funciona como:

```
Nivel 3: [Conceptos abstractos]
              ↓ predice
Nivel 2: [Secuencias de patrones]
              ↓ predice
Nivel 1: [Patrones sensoriales]
              ↓
        [Input sensorial]
```

**Principios clave:**

1. **Sparse activation:** Solo ~2% de neuronas activas simultáneamente
2. **Sequence learning:** A→B→C se aprende como pares (A,B), (B,C)
3. **Prediction:** Si veo A, pre-activo neuronas de B (estado predictivo)
4. **Surprise:** Si predicción falla → aprendizaje fuerte

### Implementación

```python
# Celda 1: Sparse Distributed Representation
class SDR:
    """
    Sparse Distributed Representation.
    Solo un pequeño % de bits están activos.
    """
    
    def __init__(self, size: int, sparsity: float = 0.02):
        """
        Args:
            size: Número total de bits
            sparsity: Fracción de bits activos (ej: 0.02 = 2%)
        """
        self.size = size
        self.sparsity = sparsity
        self.num_active = int(size * sparsity)
        self.active_indices = set()
    
    def set_random(self):
        """Activa bits aleatorios"""
        self.active_indices = set(np.random.choice(
            self.size, 
            size=self.num_active, 
            replace=False
        ))
    
    def set_from_pattern(self, pattern: np.ndarray):
        """
        Convierte un patrón denso a SDR.
        Activa los k bits con mayor valor.
        """
        # Ordenar por valor y tomar top-k
        top_indices = np.argsort(pattern)[-self.num_active:]
        self.active_indices = set(top_indices)
    
    def to_dense(self) -> np.ndarray:
        """Convierte a representación densa (array de 0s y 1s)"""
        dense = np.zeros(self.size)
        for idx in self.active_indices:
            dense[idx] = 1
        return dense
    
    def overlap(self, other: 'SDR') -> float:
        """
        Calcula overlap con otro SDR.
        Overlap = |A ∩ B| / |A ∪ B|
        """
        intersection = len(self.active_indices & other.active_indices)
        union = len(self.active_indices | other.active_indices)
        return intersection / union if union > 0 else 0.0
    
    def __repr__(self):
        return f"SDR(size={self.size}, active={len(self.active_indices)}, sparsity={len(self.active_indices)/self.size:.2%})"

# Ejemplo
sdr1 = SDR(size=2048, sparsity=0.02)
sdr1.set_random()

sdr2 = SDR(size=2048, sparsity=0.02)
sdr2.set_random()

print(f"SDR 1: {sdr1}")
print(f"SDR 2: {sdr2}")
print(f"Overlap: {sdr1.overlap(sdr2):.4f}")  # Debería ser bajo (~0 para SDRs aleatorios)
```

```python
# Celda 2: Temporal Memory
class TemporalMemory:
    """
    Memoria temporal que aprende secuencias.
    Implementación simplificada de HTM.
    """
    
    def __init__(self, num_columns: int, cells_per_column: int = 32):
        """
        Args:
            num_columns: Número de columnas (mini-columns) corticales
            cells_per_column: Células por columna
        """
        self.num_columns = num_columns
        self.cells_per_column = cells_per_column
        self.total_cells = num_columns * cells_per_column
        
        # Estado de las células
        self.active_cells = set()       # Células actualmente activas
        self.predictive_cells = set()   # Células en estado predictivo
        self.winner_cells = set()       # Células ganadoras (para aprendizaje)
        
        # Conexiones dendríticas
        # connections[cell] = {prev_cell1: permanence1, prev_cell2: permanence2, ...}
        self.connections = {i: {} for i in range(self.total_cells)}
        
        # Hiperparámetros
        self.initial_permanence = 0.21
        self.connected_permanence = 0.50
        self.permanence_increment = 0.10
        self.permanence_decrement = 0.10
        self.activation_threshold = 13  # Mínimo de conexiones activas para predecir
        
        # Para visualización
        self.history_active = []
        self.history_predictive = []
    
    def reset(self):
        """Resetea el estado (sin olvidar conexiones aprendidas)"""
        self.active_cells.clear()
        self.predictive_cells.clear()
        self.winner_cells.clear()
    
    def compute(self, active_columns: set, learn: bool = True):
        """
        Procesa un paso de tiempo.
        
        Args:
            active_columns: Conjunto de columnas activas en este paso
            learn: Si True, actualiza conexiones sinápticas
        """
        # 1. Activar células
        prev_predictive = self.predictive_cells.copy()
        self.active_cells.clear()
        self.winner_cells.clear()
        
        for column in active_columns:
            # Verificar si había predicción correcta
            predicted_cells_in_column = [
                c for c in prev_predictive 
                if c // self.cells_per_column == column
            ]
            
            if predicted_cells_in_column:
                # Predicción correcta: activar células predictivas
                self.active_cells.update(predicted_cells_in_column)
                self.winner_cells.update(predicted_cells_in_column[:1])  # Una ganadora
            else:
                # Sin predicción: activar todas las células de la columna (bursting)
                start_idx = column * self.cells_per_column
                cells_in_column = list(range(start_idx, start_idx + self.cells_per_column))
                self.active_cells.update(cells_in_column)
                # Escoger una célula ganadora aleatoria
                self.winner_cells.add(random.choice(cells_in_column))
        
        # 2. Predecir siguiente paso
        self.predictive_cells.clear()
        
        for cell in range(self.total_cells):
            # Contar cuántas conexiones activas tiene esta célula
            active_connections = sum(
                1 for prev_cell, perm in self.connections[cell].items()
                if prev_cell in self.active_cells and perm >= self.connected_permanence
            )
            
            # Si supera umbral, poner en estado predictivo
            if active_connections >= self.activation_threshold:
                self.predictive_cells.add(cell)
        
        # 3. Aprender conexiones (si habilitado)
        if learn:
            for winner_cell in self.winner_cells:
                # Reforzar conexiones con células activas en el paso anterior
                for prev_cell in self.active_cells:
                    if prev_cell in self.connections[winner_cell]:
                        # Incrementar permanencia de conexión existente
                        self.connections[winner_cell][prev_cell] += self.permanence_increment
                        self.connections[winner_cell][prev_cell] = min(1.0, self.connections[winner_cell][prev_cell])
                    else:
                        # Crear nueva conexión
                        if len(self.connections[winner_cell]) < 100:  # Límite de conexiones por célula
                            self.connections[winner_cell][prev_cell] = self.initial_permanence
                
                # Debilitar conexiones con células inactivas
                for prev_cell in list(self.connections[winner_cell].keys()):
                    if prev_cell not in self.active_cells:
                        self.connections[winner_cell][prev_cell] -= self.permanence_decrement
                        # Eliminar si permanencia cae por debajo de 0
                        if self.connections[winner_cell][prev_cell] <= 0:
                            del self.connections[winner_cell][prev_cell]
        
        # Guardar historial
        self.history_active.append(self.active_cells.copy())
        self.history_predictive.append(self.predictive_cells.copy())
    
    def get_active_columns(self) -> set:
        """Obtiene columnas activas"""
        return {cell // self.cells_per_column for cell in self.active_cells}
    
    def get_predictive_columns(self) -> set:
        """Obtiene columnas en estado predictivo"""
        return {cell // self.cells_per_column for cell in self.predictive_cells}
    
    def visualize_state(self, step: int):
        """Visualiza el estado de las células"""
        # Crear matriz (columnas x células_por_columna)
        state = np.zeros((self.num_columns, self.cells_per_column))
        
        for cell in self.active_cells:
            col = cell // self.cells_per_column
            cell_in_col = cell % self.cells_per_column
            state[col, cell_in_col] = 1.0  # Activa
        
        for cell in self.predictive_cells:
            col = cell // self.cells_per_column
            cell_in_col = cell % self.cells_per_column
            if cell not in self.active_cells:
                state[col, cell_in_col] = 0.5  # Predictiva
        
        plt.figure(figsize=(12, 4))
        plt.imshow(state.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(label='Estado', ticks=[0, 0.5, 1.0], 
                     format=plt.FuncFormatter(lambda x, p: ['Inactiva', 'Predictiva', 'Activa'][int(x*2)]))
        plt.xlabel('Columna')
        plt.ylabel('Célula dentro de columna')
        plt.title(f'Estado de Memoria Temporal (t={step})', fontsize=12, weight='bold')
        plt.tight_layout()
        plt.show()
```

```python
# Celda 3: Experimento 1 - Aprender secuencia simple
def experimento_secuencia_simple():
    """
    Aprender la secuencia: A → B → C → D → A → ...
    """
    print("=== Experimento: Aprender Secuencia A→B→C→D ===\n")
    
    # Crear TM con 100 columnas
    tm = TemporalMemory(num_columns=100, cells_per_column=32)
    
    # Definir patrones (conjuntos de columnas activas)
    patterns = {
        'A': set(range(0, 20)),      # Columnas 0-19
        'B': set(range(20, 40)),     # Columnas 20-39
        'C': set(range(40, 60)),     # Columnas 40-59
        'D': set(range(60, 80)),     # Columnas 60-79
    }
    
    sequence = ['A', 'B', 'C', 'D']
    
    # Entrenar
    num_epochs = 10
    prediction_accuracy = []
    
    for epoch in range(num_epochs):
        tm.reset()
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(len(sequence) * 5):  # Repetir secuencia 5 veces por época
            pattern_name = sequence[i % len(sequence)]
            active_columns = patterns[pattern_name]
            
            # Verificar predicción (antes de procesar)
            if i > 0:
                predicted_cols = tm.get_predictive_columns()
                if predicted_cols == active_columns:
                    correct_predictions += 1
                total_predictions += 1
            
            # Procesar
            tm.compute(active_columns, learn=True)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        prediction_accuracy.append(accuracy)
        print(f"Época {epoch+1}: Precisión de predicción = {accuracy:.2%}")
    
    # Visualizar aprendizaje
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), prediction_accuracy, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Época')
    plt.ylabel('Precisión de Predicción')
    plt.title('Aprendizaje de Secuencia Temporal', fontsize=14, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.show()
    
    # Test: Mostrar predicciones
    print("\n--- Test de Predicción ---")
    tm.reset()
    
    for i, pattern_name in enumerate(sequence):
        print(f"\nPaso {i+1}: Input = {pattern_name}")
        
        # Mostrar predicción ANTES de dar el input
        if i > 0:
            predicted_cols = tm.get_predictive_columns()
            # Identificar qué patrón se predijo
            for pname, pcols in patterns.items():
                if predicted_cols == pcols:
                    print(f"  → Predicción: {pname} ✓")
                    break
            else:
                print(f"  → Predicción: Desconocido (cols={predicted_cols})")
        
        # Procesar input
        tm.compute(patterns[pattern_name], learn=False)
        tm.visualize_state(i)
    
    return tm

experimento_secuencia_simple()
```

```python
# Celda 4: Place Cells y Grid Cells
class PlaceCell:
    """
    Célula de lugar (place cell).
    Se activa cuando el agente está en una posición específica.
    """
    
    def __init__(self, preferred_location: Tuple[float, float], radius: float = 2.0):
        """
        Args:
            preferred_location: (x, y) posición preferida
            radius: Radio de activación
        """
        self.preferred_x, self.preferred_y = preferred_location
        self.radius = radius
        self.activation = 0.0
        self.history = []
    
    def compute_activation(self, current_location: Tuple[float, float]) -> float:
        """
        Calcula activación según distancia a posición preferida.
        Activación = Gaussiana centrada en preferred_location
        """
        x, y = current_location
        distance = np.sqrt((x - self.preferred_x)**2 + (y - self.preferred_y)**2)
        
        # Gaussiana
        self.activation = np.exp(-distance**2 / (2 * self.radius**2))
        self.history.append(self.activation)
        
        return self.activation


class GridCell:
    """
    Célula de grilla (grid cell).
    Se activa en múltiples posiciones formando un patrón hexagonal.
    """
    
    def __init__(self, spacing: float = 5.0, orientation: float = 0.0, phase: Tuple[float, float] = (0, 0)):
        """
        Args:
            spacing: Espaciado entre picos de activación
            orientation: Orientación del grid (radianes)
            phase: Desplazamiento de fase (x, y)
        """
        self.spacing = spacing
        self.orientation = orientation
        self.phase_x, self.phase_y = phase
        self.activation = 0.0
        self.history = []
    
    def compute_activation(self, current_location: Tuple[float, float]) -> float:
        """
        Calcula activación en patrón hexagonal.
        Usa suma de 3 ondas sinusoidales con orientaciones 60° aparte.
        """
        x, y = current_location
        
        # Aplicar fase
        x = x - self.phase_x
        y = y - self.phase_y
        
        # 3 ondas sinusoidales con 60° de separación
        angles = [self.orientation, self.orientation + np.pi/3, self.orientation + 2*np.pi/3]
        
        wave_sum = 0
        for angle in angles:
            # Proyectar posición en dirección de la onda
            projection = x * np.cos(angle) + y * np.sin(angle)
            wave_sum += np.cos(2 * np.pi * projection / self.spacing)
        
        # Normalizar y aplicar umbral
        self.activation = max(0, (wave_sum + 1.5) / 4.5)  # Mapear a [0, 1]
        self.history.append(self.activation)
        
        return self.activation


class SpatialNavigationSystem:
    """
    Sistema de navegación espacial con place cells y grid cells.
    """
    
    def __init__(self, world_size: Tuple[int, int] = (20, 20)):
        self.world_width, self.world_height = world_size
        self.current_position = (world_size[0] / 2, world_size[1] / 2)
        
        # Crear place cells distribuidas uniformemente
        self.place_cells = []
        spacing = 3
        for x in range(0, world_size[0], spacing):
            for y in range(0, world_size[1], spacing):
                pc = PlaceCell((x, y), radius=2.0)
                self.place_cells.append(pc)
        
        # Crear grid cells con diferentes escalas
        self.grid_cells = []
        spacings = [3, 5, 7]  # Diferentes escalas
        for spacing in spacings:
            for phase_x in [0, spacing/2]:
                for phase_y in [0, spacing/2]:
                    gc = GridCell(spacing=spacing, orientation=0, phase=(phase_x, phase_y))
                    self.grid_cells.append(gc)
        
        print(f"Creado sistema con {len(self.place_cells)} place cells y {len(self.grid_cells)} grid cells")
    
    def move_to(self, new_position: Tuple[float, float]):
        """Mueve el agente a una nueva posición y actualiza células"""
        self.current_position = new_position
        
        # Actualizar place cells
        for pc in self.place_cells:
            pc.compute_activation(new_position)
        
        # Actualizar grid cells
        for gc in self.grid_cells:
            gc.compute_activation(new_position)
    
    def get_place_field_map(self) -> np.ndarray:
        """Crea mapa de activación de place cells"""
        map_resolution = 50
        place_map = np.zeros((map_resolution, map_resolution))
        
        for i in range(map_resolution):
            for j in range(map_resolution):
                x = (i / map_resolution) * self.world_width
                y = (j / map_resolution) * self.world_height
                
                # Sumar activación de todas las place cells
                total_activation = 0
                for pc in self.place_cells:
                    dist = np.sqrt((x - pc.preferred_x)**2 + (y - pc.preferred_y)**2)
                    activation = np.exp(-dist**2 / (2 * pc.radius**2))
                    total_activation += activation
                
                place_map[j, i] = total_activation
        
        return place_map
    
    def get_grid_field_map(self, grid_cell_idx: int = 0) -> np.ndarray:
        """Crea mapa de activación de una grid cell específica"""
        map_resolution = 100
        grid_map = np.zeros((map_resolution, map_resolution))
        
        gc = self.grid_cells[grid_cell_idx]
        
        for i in range(map_resolution):
            for j in range(map_resolution):
                x = (i / map_resolution) * self.world_width
                y = (j / map_resolution) * self.world_height
                
                # Calcular activación
                x_shifted = x - gc.phase_x
                y_shifted = y - gc.phase_y
                
                angles = [gc.orientation, gc.orientation + np.pi/3, gc.orientation + 2*np.pi/3]
                wave_sum = 0
                for angle in angles:
                    projection = x_shifted * np.cos(angle) + y_shifted * np.sin(angle)
                    wave_sum += np.cos(2 * np.pi * projection / gc.spacing)
                
                grid_map[j, i] = max(0, (wave_sum + 1.5) / 4.5)
        
        return grid_map
```

```python
# Celda 5: Experimento 2 - Navegación con place cells
def experimento_place_cells():
    """
    Simula un agente navegando y activa place cells.
    """
    print("=== Experimento: Place Cells y Grid Cells ===\n")
    
    nav_system = SpatialNavigationSystem(world_size=(20, 20))
    
    # Simular trayectoria: Cuadrado
    trajectory = []
    positions = [
        (5, 5), (15, 5), (15, 15), (5, 15), (5, 5)
    ]
    
    # Interpolar entre posiciones
    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[i + 1]
        steps = 20
        
        for t in range(steps):
            alpha = t / steps
            x = start[0] + alpha * (end[0] - start[0])
            y = start[1] + alpha * (end[1] - start[1])
            trajectory.append((x, y))
            nav_system.move_to((x, y))
    
    # Visualizar
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Trayectoria
    traj_array = np.array(trajectory)
    axes[0, 0].plot(traj_array[:, 0], traj_array[:, 1], 'b-', linewidth=2)
    axes[0, 0].plot(traj_array[0, 0], traj_array[0, 1], 'go', markersize=15, label='Start')
    axes[0, 0].plot(traj_array[-1, 0], traj_array[-1, 1], 'ro', markersize=15, label='End')
    axes[0, 0].set_xlim([0, 20])
    axes[0, 0].set_ylim([0, 20])
    axes[0, 0].set_title('Trayectoria del Agente', fontsize=12, weight='bold')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Mapa de place cells
    place_map = nav_system.get_place_field_map()
    im1 = axes[0, 1].imshow(place_map, cmap='hot', origin='lower', extent=[0, 20, 0, 20])
    axes[0, 1].plot(traj_array[:, 0], traj_array[:, 1], 'c-', linewidth=2, alpha=0.5)
    axes[0, 1].set_title('Place Fields (Cobertura)', fontsize=12, weight='bold')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 3-5. Grid cells (diferentes escalas)
    for idx in range(3):
        grid_map = nav_system.get_grid_field_map(grid_cell_idx=idx*4)
        im = axes[0 if idx < 2 else 1, 2 if idx == 0 else (0 if idx == 1 else 1)].imshow(
            grid_map, cmap='viridis', origin='lower', extent=[0, 20, 0, 20]
        )
        axes[0 if idx < 2 else 1, 2 if idx == 0 else (0 if idx == 1 else 1)].plot(
            traj_array[:, 0], traj_array[:, 1], 'r-', linewidth=1.5, alpha=0.5
        )
        axes[0 if idx < 2 else 1, 2 if idx == 0 else (0 if idx == 1 else 1)].set_title(
            f'Grid Cell {idx+1} (spacing={nav_system.grid_cells[idx*4].spacing})', 
            fontsize=11, weight='bold'
        )
        axes[0 if idx < 2 else 1, 2 if idx == 0 else (0 if idx == 1 else 1)].set_xlabel('X')
        axes[0 if idx < 2 else 1, 2 if idx == 0 else (0 if idx == 1 else 1)].set_ylabel('Y')
        plt.colorbar(im, ax=axes[0 if idx < 2 else 1, 2 if idx == 0 else (0 if idx == 1 else 1)])
    
    # 6. Activación temporal de place cells
    # Seleccionar 5 place cells representativas
    selected_pcs = [nav_system.place_cells[i] for i in [0, 10, 20, 30, 40]]
    
    for i, pc in enumerate(selected_pcs):
        axes[1, 2].plot(pc.history, label=f'PC {i+1} @ ({pc.preferred_x:.1f},{pc.preferred_y:.1f})')
    
    axes[1, 2].set_xlabel('Paso de tiempo')
    axes[1, 2].set_ylabel('Activación')
    axes[1, 2].set_title('Activación Temporal de Place Cells', fontsize=11, weight='bold')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Observa:")
    print("1. Place cells: Activación localizada en posiciones específicas")
    print("2. Grid cells: Patrón hexagonal que cubre todo el espacio")
    print("3. Activación temporal: Cada place cell 'dispara' cuando el agente pasa por su zona")
    print("4. Grid cells con diferentes escalas cubren distintas resoluciones espaciales")
    
    return nav_system

experimento_place_cells()
```

```python
# Celda 6: Memory Replay
def experimento_memory_replay():
    """
    Simula memory replay: Reproducir una secuencia aprendida sin input sensorial.
    """
    print("=== Experimento: Memory Replay ===\n")
    
    # Crear TM y aprender secuencia
    tm = TemporalMemory(num_columns=50, cells_per_column=32)
    
    # Secuencia de lugares: A → B → C → D
    patterns = {
        'A': set(range(0, 10)),
        'B': set(range(10, 20)),
        'C': set(range(20, 30)),
        'D': set(range(30, 40)),
    }
    
    sequence = ['A', 'B', 'C', 'D']
    
    # Entrenar bien
    print("Entrenando secuencia...")
    for epoch in range(20):
        tm.reset()
        for _ in range(5):  # Repetir secuencia 5 veces por época
            for pattern_name in sequence:
                tm.compute(patterns[pattern_name], learn=True)
    
    print("Entrenamiento completo.\n")
    
    # Test 1: Navegación normal (con input sensorial)
    print("--- Test 1: Navegación Normal (con sensores) ---")
    tm.reset()
    activations_normal = []
    
    for i, pattern_name in enumerate(sequence * 2):  # Repetir 2 veces
        tm.compute(patterns[pattern_name], learn=False)
        activations_normal.append(tm.get_active_columns())
    
    # Test 2: Memory Replay (sin input sensorial después del primer paso)
    print("--- Test 2: Memory Replay (sin sensores) ---")
    tm.reset()
    activations_replay = []
    
    # Solo dar el primer input (A), luego dejar que la red "imagine"
    tm.compute(patterns['A'], learn=False)
    activations_replay.append(tm.get_active_columns())
    
    print("  Paso 1: Input = A (sensorial)")
    
    for i in range(7):  # Intentar 7 pasos más
        # NO DAR INPUT, solo dejar que las predicciones se activen
        # Simular: Las columnas predictivas se convierten en activas
        predicted_cols = tm.get_predictive_columns()
        
        if not predicted_cols:
            print(f"  Paso {i+2}: Predicción vacía → Memory replay terminado")
            break
        
        tm.compute(predicted_cols, learn=False)
        activations_replay.append(tm.get_active_columns())
        
        # Identificar qué patrón se reprodujo
        for pname, pcols in patterns.items():
            if predicted_cols == pcols:
                print(f"  Paso {i+2}: Reproduciendo {pname} (sin input sensorial!)")
                break
    
    # Visualizar
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Navegación normal
    normal_matrix = np.zeros((len(activations_normal), tm.num_columns))
    for t, active_cols in enumerate(activations_normal):
        for col in active_cols:
            normal_matrix[t, col] = 1
    
    axes[0].imshow(normal_matrix.T, cmap='hot', aspect='auto')
    axes[0].set_title('Navegación Normal (con input sensorial)', fontsize=13, weight='bold')
    axes[0].set_xlabel('Paso de tiempo')
    axes[0].set_ylabel('Columna')
    axes[0].set_yticks([5, 15, 25, 35])
    axes[0].set_yticklabels(['A', 'B', 'C', 'D'])
    
    # Plot 2: Memory replay
    replay_matrix = np.zeros((len(activations_replay), tm.num_columns))
    for t, active_cols in enumerate(activations_replay):
        for col in active_cols:
            replay_matrix[t, col] = 1
    
    axes[1].imshow(replay_matrix.T, cmap='hot', aspect='auto')
    axes[1].set_title('Memory Replay (sin input sensorial después de t=0)', fontsize=13, weight='bold')
    axes[1].set_xlabel('Paso de tiempo')
    axes[1].set_ylabel('Columna')
    axes[1].set_yticks([5, 15, 25, 35])
    axes[1].set_yticklabels(['A', 'B', 'C', 'D'])
    axes[1].axvline(0.5, color='cyan', linestyle='--', linewidth=2, label='Input sensorial')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n¡Observa cómo la red reproduce la secuencia A→B→C→D sin input externo!")
    print("Esto es similar a lo que hacen las ratas durante el sueño (memory replay)")
    
    return tm

experimento_memory_replay()
```

### Resumen del Notebook 3

```markdown
## ✅ Logros del Notebook 3

1. **Sparse Distributed Representations:**
   - SDR con ~2% de activación
   - Overlap para medir similaridad

2. **Temporal Memory (HTM):**
   - Aprendizaje de secuencias temporales
   - Predicción del siguiente estado
   - Estado predictivo vs. activo

3. **Place Cells y Grid Cells:**
   - Células de lugar (activación localizada)
   - Células de grilla (patrón hexagonal)
   - Sistema de navegación espacial

4. **Memory Replay:**
   - Reproducir secuencias sin input sensorial
   - Simulación de "sueño" o "imaginación"

5. **Aplicaciones:**
   - Predicción de secuencias
   - Navegación espacial
   - Planificación (reproducir rutas)

## 🚀 Próximos Pasos (Notebook 4)

- **UI interactiva** con ipycanvas
- **Dibujar neuronas** como píxeles
- **Robot simulado** navegando en grid
- **Experimentos** de aprendizaje motor
```

---

## 📘 Notebook 4: UI Interactiva y Robótica

### Objetivos

1. **Canvas interactivo** con `ipycanvas` para dibujar neuronas
2. **Controles UI** con `ipywidgets` (play/pause, sliders)
3. **Robot simulado** navegando con CNA
4. **Experimentos** de aprendizaje sensorimotor
5. **Integración completa** de todos los componentes

### Implementación

```python
# Celda 1: UI Interactiva Completa
from ipycanvas import Canvas, hold_canvas
import ipywidgets as widgets
from IPython.display import display
import asyncio

class InteractiveCNA:
    """
    Interfaz interactiva para el Connectionist Neural Automaton.
    Permite dibujar, ejecutar, y visualizar en tiempo real.
    """
    
    def __init__(self, grid_size=(64, 64), cell_size=10):
        self.grid_width, self.grid_height = grid_size
        self.cell_size = cell_size
        
        # Canvas dimensions
        self.canvas_width = self.grid_width * cell_size
        self.canvas_height = self.grid_height * cell_size
        
        # Crear CNA
        self.cna = CNA_ConSOM(self.grid_width, self.grid_height)
        
        # Estado de animación
        self.running = False
        self.speed = 10  # Steps per second
        self.current_step = 0
        
        # Estado de dibujo
        self.drawing = False
        self.brush_size = 2
        self.brush_value = 1.0
        self.current_region = config.REGIONES['INTERNA']
        
        # Crear UI
        self._create_canvas()
        self._create_controls()
        self._setup_layout()
        
        # Render inicial
        self._render()
    
    def _create_canvas(self):
        """Crea el canvas principal"""
        self.canvas = Canvas(width=self.canvas_width, height=self.canvas_height)
        
        # Event handlers
        self.canvas.on_mouse_down(self._on_mouse_down)
        self.canvas.on_mouse_move(self._on_mouse_move)
        self.canvas.on_mouse_up(self._on_mouse_up)
    
    def _create_controls(self):
        """Crea controles de la UI"""
        # Botones
        self.btn_play = widgets.Button(description='▶ Play', button_style='success')
        self.btn_step = widgets.Button(description='⏭ Step', button_style='info')
        self.btn_reset = widgets.Button(description='🔄 Reset', button_style='warning')
        self.btn_clear = widgets.Button(description='🗑 Clear', button_style='danger')
        
        self.btn_play.on_click(self._toggle_play)
        self.btn_step.on_click(self._step_once)
        self.btn_reset.on_click(self._reset)
        self.btn_clear.on_click(self._clear)
        
        # Sliders
        self.slider_speed = widgets.IntSlider(
            value=10, min=1, max=60, step=1,
            description='Speed (fps):', style={'description_width': 'initial'}
        )
        self.slider_speed.observe(self._on_speed_change, 'value')
        
        self.slider_brush = widgets.IntSlider(
            value=2, min=1, max=10, step=1,
            description='Brush Size:', style={'description_width': 'initial'}
        )
        self.slider_brush.observe(self._on_brush_change, 'value')
        
        self.slider_learning_rate = widgets.FloatSlider(
            value=config.COEF_SINAPSIS_ENTRENAMIENTO, min=0.0, max=0.5, step=0.01,
            description='Learning Rate:', style={'description_width': 'initial'}
        )
        
        # Dropdown región
        self.dropdown_region = widgets.Dropdown(
            options=[('ENTRADA', 0), ('SALIDA', 1), ('INTERNA', 2), ('DOLOR', 3)],
            value=2,
            description='Draw Region:',
            style={'description_width': 'initial'}
        )
        self.dropdown_region.observe(self._on_region_change, 'value')
        
        # Checkbox
        self.checkbox_learning = widgets.Checkbox(
            value=True, description='Enable Learning'
        )
        self.checkbox_learning.observe(self._on_learning_toggle, 'value')
        
        # Label para info
        self.label_info = widgets.Label(value=f'Step: 0 | Active neurons: 0')
    
    def _setup_layout(self):
        """Organiza el layout de la UI"""
        # Fila de botones
        buttons = widgets.HBox([self.btn_play, self.btn_step, self.btn_reset, self.btn_clear])
        
        # Controles
        controls = widgets.VBox([
            self.slider_speed,
            self.slider_brush,
            self.slider_learning_rate,
            self.dropdown_region,
            self.checkbox_learning,
            self.label_info
        ])
        
        # Layout principal
        self.layout = widgets.VBox([
            widgets.HTML("<h2>🧠 Connectionist Neural Automaton - Interactive UI</h2>"),
            buttons,
            self.canvas,
            controls
        ])
    
    def display(self):
        """Muestra la UI"""
        display(self.layout)
    
    def _render(self):
        """Renderiza el estado actual del CNA"""
        with hold_canvas(self.canvas):
            self.canvas.clear()
            
            # Dibujar grid
            for y in range(self.grid_height):
                for x in range(self.grid_width):
                    neuron = self.cna.grid[y][x]
                    
                    # Color según activación y región
                    if neuron.region == config.REGIONES['ENTRADA']:
                        base_color = (100, 149, 237)  # Azul (entrada)
                    elif neuron.region == config.REGIONES['SALIDA']:
                        base_color = (220, 20, 60)    # Rojo (salida)
                    elif neuron.region == config.REGIONES['DOLOR']:
                        base_color = (255, 140, 0)    # Naranja (dolor)
                    else:
                        base_color = (200, 200, 200)  # Gris (interna)
                    
                    # Modular por activación
                    intensity = neuron.valor
                    r = int(base_color[0] * (0.2 + 0.8 * intensity))
                    g = int(base_color[1] * (0.2 + 0.8 * intensity))
                    b = int(base_color[2] * (0.2 + 0.8 * intensity))
                    
                    # Dibujar célula
                    self.canvas.fill_style = f'rgb({r},{g},{b})'
                    px = x * self.cell_size
                    py = y * self.cell_size
                    self.canvas.fill_rect(px, py, self.cell_size-1, self.cell_size-1)
            
            # Actualizar info
            active_count = sum(1 for n in self.cna.neuronas if n.valor > 0.5)
            self.label_info.value = f'Step: {self.current_step} | Active neurons: {active_count}/{len(self.cna.neuronas)}'
    
    def _on_mouse_down(self, x, y):
        """Handler para mouse down"""
        self.drawing = True
        self._paint_at(x, y)
    
    def _on_mouse_move(self, x, y):
        """Handler para mouse move"""
        if self.drawing:
            self._paint_at(x, y)
    
    def _on_mouse_up(self, x, y):
        """Handler para mouse up"""
        self.drawing = False
    
    def _paint_at(self, canvas_x, canvas_y):
        """Pinta neuronas en la posición del mouse"""
        # Convertir coordenadas de canvas a grid
        grid_x = int(canvas_x / self.cell_size)
        grid_y = int(canvas_y / self.cell_size)
        
        # Pintar con brush size
        for dy in range(-self.brush_size, self.brush_size + 1):
            for dx in range(-self.brush_size, self.brush_size + 1):
                nx, ny = grid_x + dx, grid_y + dy
                
                # Verificar límites
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    neuron = self.cna.grid[ny][nx]
                    neuron.valor = self.brush_value
                    neuron.region = self.current_region
        
        self._render()
    
    def _toggle_play(self, btn):
        """Toggle play/pause"""
        self.running = not self.running
        
        if self.running:
            self.btn_play.description = '⏸ Pause'
            self.btn_play.button_style = 'warning'
            asyncio.ensure_future(self._animation_loop())
        else:
            self.btn_play.description = '▶ Play'
            self.btn_play.button_style = 'success'
    
    async def _animation_loop(self):
        """Loop de animación"""
        while self.running:
            # Step
            config.COEF_SINAPSIS_ENTRENAMIENTO = self.slider_learning_rate.value
            self.cna.learning_enabled = self.checkbox_learning.value
            
            self.cna.step_with_lateral()
            self.current_step += 1
            
            # Render
            self._render()
            
            # Delay según speed
            await asyncio.sleep(1.0 / self.speed)
    
    def _step_once(self, btn):
        """Ejecuta un paso"""
        config.COEF_SINAPSIS_ENTRENAMIENTO = self.slider_learning_rate.value
        self.cna.learning_enabled = self.checkbox_learning.value
        
        self.cna.step_with_lateral()
        self.current_step += 1
        self._render()
    
    def _reset(self, btn):
        """Resetea el CNA"""
        self.running = False
        self.btn_play.description = '▶ Play'
        self.btn_play.button_style = 'success'
        
        self.cna.reset()
        self.current_step = 0
        self._render()
    
    def _clear(self, btn):
        """Limpia el canvas (resetea valores pero mantiene pesos)"""
        for neuron in self.cna.neuronas:
            neuron.valor = 0.0
            neuron.activa = False
        
        self._render()
    
    def _on_speed_change(self, change):
        """Handler para cambio de velocidad"""
        self.speed = change['new']
    
    def _on_brush_change(self, change):
        """Handler para cambio de brush size"""
        self.brush_size = change['new']
    
    def _on_region_change(self, change):
        """Handler para cambio de región"""
        self.current_region = change['new']
    
    def _on_learning_toggle(self, change):
        """Handler para toggle de aprendizaje"""
        self.cna.learning_enabled = change['new']

# Crear y mostrar UI
print("Creando UI interactiva...")
ui = InteractiveCNA(grid_size=(48, 48), cell_size=12)
ui.display()

print("\n✅ UI lista!")
print("📝 Instrucciones:")
print("  • Click y arrastra para dibujar neuronas activas")
print("  • Usa Play para ver la evolución automática")
print("  • Step para avanzar un paso a la vez")
print("  • Ajusta sliders para controlar comportamiento")
```

```python
# Celda 2: Robot Simulado
class SimpleRobot:
    """
    Robot simple que navega en un grid 2D.
    """
    
    def __init__(self, grid_size=(20, 20)):
        self.width, self.height = grid_size
        self.position = (grid_size[0] // 2, grid_size[1] // 2)
        self.orientation = 0  # 0=N, 1=E, 2=S, 3=W
        
        # Mundo
        self.world = np.zeros(grid_size)
        self.goal_position = None
        self.obstacles = set()
        
        # Sensores (8 direcciones)
        self.sensor_readings = np.zeros(8)
        
        # Historial
        self.trajectory = [self.position]
    
    def set_goal(self, position: Tuple[int, int]):
        """Establece la posición objetivo"""
        self.goal_position = position
        self.world[position] = 2.0  # Valor alto para objetivo
    
    def add_obstacle(self, position: Tuple[int, int]):
        """Añade un obstáculo"""
        self.obstacles.add(position)
        self.world[position] = -1.0  # Valor negativo para obstáculos
    
    def sense(self) -> np.ndarray:
        """
        Lee sensores (distancia a objetivo y obstáculos en 8 direcciones).
        """
        directions = [
            (-1, 0),   # N
            (-1, 1),   # NE
            (0, 1),    # E
            (1, 1),    # SE
            (1, 0),    # S
            (1, -1),   # SW
            (0, -1),   # W
            (-1, -1),  # NW
        ]
        
        x, y = self.position
        
        for i, (dx, dy) in enumerate(directions):
            # Raycast en esta dirección
            distance_to_goal = float('inf')
            distance_to_obstacle = float('inf')
            
            for step in range(1, max(self.width, self.height)):
                nx, ny = x + dx * step, y + dy * step
                
                # Fuera de límites
                if not (0 <= nx < self.height and 0 <= ny < self.width):
                    break
                
                # Objetivo
                if (nx, ny) == self.goal_position:
                    distance_to_goal = step
                    break
                
                # Obstáculo
                if (nx, ny) in self.obstacles:
                    distance_to_obstacle = step
                    break
            
            # Sensor reading: positivo si objetivo cerca, negativo si obstáculo cerca
            if distance_to_goal < distance_to_obstacle:
                self.sensor_readings[i] = 1.0 / distance_to_goal if distance_to_goal < 10 else 0.0
            else:
                self.sensor_readings[i] = -1.0 / distance_to_obstacle if distance_to_obstacle < 5 else 0.0
        
        return self.sensor_readings
    
    def move(self, action: str):
        """
        Ejecuta una acción: 'forward', 'turn_left', 'turn_right'
        """
        x, y = self.position
        
        if action == 'forward':
            # Mover según orientación
            direction_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
            dx, dy = direction_map[self.orientation]
            nx, ny = x + dx, y + dy
            
            # Verificar límites y obstáculos
            if (0 <= nx < self.height and 0 <= ny < self.width and 
                (nx, ny) not in self.obstacles):
                self.position = (nx, ny)
                self.trajectory.append(self.position)
        
        elif action == 'turn_left':
            self.orientation = (self.orientation - 1) % 4
        
        elif action == 'turn_right':
            self.orientation = (self.orientation + 1) % 4
    
    def at_goal(self) -> bool:
        """Verifica si llegó al objetivo"""
        return self.position == self.goal_position
    
    def get_state_vector(self) -> np.ndarray:
        """Estado completo: posición + sensores"""
        # Normalizar posición
        pos_x = self.position[1] / self.width
        pos_y = self.position[0] / self.height
        
        # Normalizar sensores a [0, 1]
        sensors_norm = (self.sensor_readings + 1) / 2
        
        return np.concatenate([[pos_x, pos_y], sensors_norm])
    
    def visualize(self, ax=None):
        """Visualiza el mundo y el robot"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Dibujar mundo
        world_vis = self.world.copy()
        world_vis[self.position] = 1.0  # Robot
        
        ax.imshow(world_vis, cmap='RdYlGn', vmin=-1, vmax=2)
        
        # Trayectoria
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            ax.plot(traj[:, 1], traj[:, 0], 'b-', linewidth=2, alpha=0.6)
        
        # Robot
        rx, ry = self.position
        ax.plot(ry, rx, 'bo', markersize=15)
        
        # Orientación (flecha)
        direction_map = {0: (0, -0.5), 1: (0.5, 0), 2: (0, 0.5), 3: (-0.5, 0)}
        dx, dy = direction_map[self.orientation]
        ax.arrow(ry, rx, dy, dx, head_width=0.3, head_length=0.3, fc='blue', ec='blue')
        
        ax.set_xlim([-0.5, self.width - 0.5])
        ax.set_ylim([self.height - 0.5, -0.5])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return ax


class RobotBrainInterface:
    """
    Conecta un CNA con un robot simulado.
    El CNA aprende a navegar el robot hacia el objetivo.
    """
    
    def __init__(self, robot: SimpleRobot, cna_size=(32, 32)):
        self.robot = robot
        self.cna = CNA_ConSOM(*cna_size)
        self.cna_size = cna_size
        
        # Mapear regiones
        # ENTRADA: Sensores del robot
        # SALIDA: Comandos motores (forward, turn_left, turn_right)
    
    def sensors_to_input(self, sensors: np.ndarray):
        """Convierte lecturas de sensores a patrón de activación"""
        # Normalizar sensores a [0, 1]
        sensors_norm = (sensors + 1) / 2
        
        # Activar región ENTRADA según sensores
        input_pattern = np.zeros(self.cna_size[0])
        
        # Mapear 8 sensores a ancho de región ENTRADA
        for i, sensor_val in enumerate(sensors_norm):
            # Activar neurona correspondiente
            neuron_idx = int((i / 8) * self.cna_size[0])
            input_pattern[neuron_idx] = sensor_val
        
        return input_pattern
    
    def output_to_action(self) -> str:
        """Lee región SALIDA y decide acción"""
        output = self.cna.get_output_region()
        
        # Dividir output en 3 zonas: forward, turn_left, turn_right
        third = len(output) // 3
        
        forward_activation = output[:third].mean()
        left_activation = output[third:2*third].mean()
        right_activation = output[2*third:].mean()
        
        # Acción con mayor activación
        activations = {
            'forward': forward_activation,
            'turn_left': left_activation,
            'turn_right': right_activation
        }
        
        return max(activations, key=activations.get)
    
    def step(self) -> Tuple[str, float]:
        """
        Un paso de control:
        1. Leer sensores
        2. Activar CNA
        3. Leer salida
        4. Ejecutar acción
        5. Calcular recompensa
        """
        # 1. Sensores
        sensors = self.robot.sense()
        
        # 2. Activar región ENTRADA
        input_pattern = self.sensors_to_input(sensors)
        self.cna.set_input_region(input_pattern)
        
        # 3. Propagar activación (10 pasos internos)
        for _ in range(10):
            self.cna.step_with_lateral()
        
        # 4. Leer salida y ejecutar
        action = self.output_to_action()
        prev_position = self.robot.position
        self.robot.move(action)
        
        # 5. Calcular recompensa
        # Recompensa: Acercarse al objetivo
        if self.robot.at_goal():
            reward = 10.0
        else:
            # Distancia antes y después
            goal = self.robot.goal_position
            dist_before = abs(prev_position[0] - goal[0]) + abs(prev_position[1] - goal[1])
            dist_after = abs(self.robot.position[0] - goal[0]) + abs(self.robot.position[1] - goal[1])
            
            reward = dist_before - dist_after  # Positivo si se acercó
        
        return action, reward
    
    def train_episode(self, max_steps=100, visualize=False):
        """Entrena un episodio completo"""
        self.robot.trajectory = [self.robot.position]
        total_reward = 0
        actions_taken = []
        
        for step in range(max_steps):
            action, reward = self.step()
            total_reward += reward
            actions_taken.append(action)
            
            # Reforzar o debilitar pesos según recompensa
            if reward > 0:
                # Reforzar conexiones activas
                config.COEF_SINAPSIS_ENTRENAMIENTO = 0.1
            else:
                # Debilitar
                config.COEF_SINAPSIS_ENTRENAMIENTO = -0.05
            
            # Si llegó al objetivo, terminar
            if self.robot.at_goal():
                print(f"¡Objetivo alcanzado en {step+1} pasos!")
                break
        
        if visualize:
            fig, ax = plt.subplots(figsize=(8, 8))
            self.robot.visualize(ax)
            ax.set_title(f'Episodio - Recompensa total: {total_reward:.2f}', fontsize=12)
            plt.show()
        
        return total_reward, len(actions_taken)
```

```python
# Celda 3: Experimento - Robot aprendiendo navegación
def experimento_robot_navegacion():
    """
    Entrena un robot a navegar hacia un objetivo usando CNA.
    """
    print("=== Experimento: Robot con Cerebro CNA ===\n")
    
    # Crear mundo
    robot = SimpleRobot(grid_size=(20, 20))
    robot.set_goal((18, 18))  # Esquina inferior derecha
    
    # Añadir obstáculos
    for i in range(5, 15):
        robot.add_obstacle((i, 10))  # Pared vertical
    
    # Crear interfaz
    interface = RobotBrainInterface(robot, cna_size=(32, 32))
    
    # Entrenar varios episodios
    num_episodes = 20
    rewards = []
    steps_taken = []
    
    for episode in range(num_episodes):
        # Resetear robot
        robot.position = (1, 1)  # Esquina superior izquierda
        robot.orientation = 0
        robot.trajectory = [robot.position]
        
        # Ejecutar episodio
        total_reward, steps = interface.train_episode(
            max_steps=150, 
            visualize=(episode % 5 == 0)  # Visualizar cada 5 episodios
        )
        
        rewards.append(total_reward)
        steps_taken.append(steps)
        
        print(f"Episodio {episode+1}: Recompensa={total_reward:.2f}, Pasos={steps}")
    
    # Plot aprendizaje
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(range(1, num_episodes+1), rewards, marker='o', linewidth=2)
    axes[0].set_xlabel('Episodio')
    axes[0].set_ylabel('Recompensa Total')
    axes[0].set_title('Progreso de Aprendizaje', fontsize=13, weight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(range(1, num_episodes+1), steps_taken, marker='s', linewidth=2, color='orange')
    axes[1].set_xlabel('Episodio')
    axes[1].set_ylabel('Pasos hasta objetivo')
    axes[1].set_title('Eficiencia de Navegación', fontsize=13, weight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Observa cómo:")
    print("  1. Primeros episodios: El robot explora aleatoriamente")
    print("  2. Episodios medios: Empieza a encontrar rutas")
    print("  3. Últimos episodios: Navegación más eficiente")
    print("  4. Los pesos sinápticos aprenden la política de navegación")
    
    return interface

experimento_robot_navegacion()
```

### Guardar y Compartir

```python
# Celda 4: Exportar animación
from matplotlib.animation import FuncAnimation, PillowWriter

def crear_animacion_cna(cna: ConnessionistNeuralAutomaton, num_steps=100, interval=50):
    """
    Crea una animación GIF de la evolución del CNA.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    states = []
    for _ in range(num_steps):
        states.append(cna.get_state().copy())
        cna.step_with_lateral()
    
    im = ax.imshow(states[0], cmap='hot', vmin=0, vmax=1, animated=True)
    ax.axis('off')
    
    def update(frame):
        im.set_array(states[frame])
        ax.set_title(f'CNA Evolution - Step {frame}', fontsize=14, weight='bold')
        return [im]
    
    anim = FuncAnimation(fig, update, frames=num_steps, interval=interval, blit=True)
    
    # Guardar como GIF
    writer = PillowWriter(fps=20)
    anim.save('cna_evolution.gif', writer=writer)
    print("✅ Animación guardada como 'cna_evolution.gif'")
    
    plt.close()
    return anim

# Ejemplo (descomentar para usar)
# cna_demo = CNA_ConSOM(48, 48)
# # Activar algunos puntos
# for _ in range(20):
#     x, y = random.randint(0, 47), random.randint(0, 15)
#     cna_demo.grid[y][x].valor = 1.0
# 
# crear_animacion_cna(cna_demo, num_steps=100)
```

### Resumen del Notebook 4

```markdown
## ✅ Logros del Notebook 4

1. **UI Interactiva Completa:**
   - Canvas para dibujar neuronas con mouse
   - Controles (play/pause, step, reset, clear)
   - Sliders para velocidad, brush, learning rate
   - Visualización en tiempo real

2. **Robot Simulado:**
   - Navegación en grid 2D
   - 8 sensores direccionales
   - 3 acciones (forward, turn_left, turn_right)
   - Detección de objetivos y obstáculos

3. **Integración CNA-Robot:**
   - Sensores → Región ENTRADA
   - Región SALIDA → Acciones motoras
   - Aprendizaje por refuerzo (Hebbian + reward)

4. **Experimentos:**
   - Robot aprendiendo navegación
   - Evitar obstáculos
   - Encontrar objetivo

5. **Exportación:**
   - Guardar estados como pickle
   - Crear animaciones GIF

## 🎉 ¡Proyecto Completo!

Has implementado un **Connectionist Neural Automaton** completo con:

✅ Autómata celular neuronal base  
✅ Self-organizing maps (Kohonen)  
✅ Memoria temporal (HTM)  
✅ UI interactiva (ipycanvas)  
✅ Robótica (navegación sensorimotor)

### Próximos Pasos Avanzados:

1. **Conectar con transformers** (embeddings de lenguaje)
2. **Jerarquía de niveles** (HTM multi-capa)
3. **Entornos 3D** (PyBullet, MuJoCo)
4. **Clustering dinámico** (nuevas regiones emergentes)
5. **Integración con LLMs** (instrucciones en lenguaje natural → acciones)
```

---

## 🌐 Compartir en la Web

### Opción 1: Google Colab (Recomendado)

**Ventajas:**
- GPU gratuita (T4, P100)
- No requiere instalación
- Fácil de compartir

**Pasos:**

1. **Subir notebooks a tu repositorio:**
   ```bash
   cd /ruta/a/tu/proyecto
   git add notebooks/*.ipynb
   git commit -m "Add CNA notebooks"
   git push origin master
   ```

2. **Crear enlaces de Colab:**
   - Reemplaza `https://github.com/` con `https://colab.research.google.com/github/`
   - Ejemplo:
     ```
     GitHub: https://github.com/tuusuario/CNA_Project/blob/master/notebooks/01_Automata_Base.ipynb
     Colab:  https://colab.research.google.com/github/tuusuario/CNA_Project/blob/master/notebooks/01_Automata_Base.ipynb
     ```

3. **Añadir badges al README:**
   ```markdown
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tuusuario/CNA_Project/blob/master/notebooks/01_Automata_Base.ipynb)
   ```

**Configurar GPU en Colab:**
```python
# Primera celda del notebook
!nvidia-smi  # Verificar GPU disponible

# Runtime > Change runtime type > Hardware accelerator > GPU
```

### Opción 2: Binder

**Ventajas:**
- 100% open-source
- Sin necesidad de cuenta
- Reproducibilidad perfecta

**Pasos:**

1. **Añadir `environment.yml` o `requirements.txt`** (ya incluido en el proyecto)

2. **Ir a mybinder.org** y crear enlace:
   - Repository: `https://github.com/tuusuario/CNA_Project`
   - Branch: `master`
   - Path to a notebook: `notebooks/01_Automata_Base.ipynb`

3. **Badge:**
   ```markdown
   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tuusuario/CNA_Project/HEAD?labpath=notebooks%2F01_Automata_Base.ipynb)
   ```

### Opción 3: GitHub Pages (Solo visualización)

**Para mostrar notebooks renderizados:**

1. **Usar nbviewer:**
   ```markdown
   [![View on nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/tuusuario/CNA_Project/blob/master/notebooks/01_Automata_Base.ipynb)
   ```

2. **O convertir a HTML:**
   ```bash
   jupyter nbconvert --to html notebooks/01_Automata_Base.ipynb
   ```

---

## 📚 Apéndices

### A. Conceptos Clave

**1. Conexionismo**
- Paradigma que modela cognición como redes de unidades simples interconectadas
- Sin procesador central (vs. cognitivismo clásico)
- Aprendizaje distribuido

**2. Autómata Celular**
- Sistema discreto de células en grid
- Cada célula tiene estado finito
- Reglas locales → comportamiento global emergente

**3. Hebbian Learning**
- "Neurons that fire together, wire together"
- Regla: Δw = η · pre · post
- Fundamento del aprendizaje asociativo

**4. Sparse Distributed Representation (SDR)**
- Solo ~2% de neuronas activas
- Alta capacidad de representación
- Robusto a ruido

**5. Predictive Coding**
- Cerebro como máquina de predicción
- Minimiza error de predicción
- Fundamental en HTM

### B. Configuración de Colab

```python
# Celda de configuración para Colab
import sys

# Verificar si estamos en Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("🔧 Configurando Google Colab...")
    
    # Verificar GPU
    !nvidia-smi
    
    # Instalar dependencias (si no están en requirements.txt)
    !pip install -q torch torchvision ipycanvas ipywidgets
    
    # Habilitar widgets
    from google.colab import output
    output.enable_custom_widget_manager()
    
    print("✅ Configuración completa!")
else:
    print("💻 Ejecutando localmente")
```

### C. Troubleshooting

**Problema: ipycanvas no funciona en Jupyter Lab**
```bash
# Solución: Instalar extensión
jupyter labextension install @jupyter-widgets/jupyterlab-manager ipycanvas
```

**Problema: GPU no detectada en Colab**
```python
# Verificar disponibilidad
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Si no está disponible: Runtime > Change runtime type > GPU
```

**Problema: Memoria insuficiente**
```python
# Reducir tamaño del grid
cna = CNA_ConSOM(32, 32)  # En vez de 64x64

# O reducir batch size en entrenamiento
```

### D. Optimizaciones Avanzadas

**1. Usar torch.compile() (PyTorch 2.0+)**
```python
import torch

@torch.compile(mode="reduce-overhead")
def cna_step_optimized(states, weights):
    return torch.nn.functional.conv2d(states, weights)

# 100x más rápido que loops Python
```

**2. Mixed Precision**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**3. Paralelizar con multiprocessing**
```python
from multiprocessing import Pool

def train_episode(episode_id):
    # Entrenar un episodio
    return reward

# Entrenar 10 episodios en paralelo
with Pool(10) as p:
    rewards = p.map(train_episode, range(10))
```

### E. Referencias y Lecturas

**Papers Fundamentales:**

1. **Temporal Memory:**
   - Hawkins, J., & Ahmad, S. (2016). "Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex"

2. **Place & Grid Cells:**
   - O'Keefe, J., & Dostrovsky, J. (1971). "The hippocampus as a spatial map"
   - Hafting, T. et al. (2005). "Microstructure of a spatial map in the entorhinal cortex"

3. **Self-Organizing Maps:**
   - Kohonen, T. (1990). "The self-organizing map"

4. **Neural Cellular Automata:**
   - Mordvintsev, A. et al. (2020). "Growing Neural Cellular Automata"

5. **Predictive Coding:**
   - Rao, R. P., & Ballard, D. H. (1999). "Predictive coding in the visual cortex"

**Libros:**

1. **On Intelligence** - Jeff Hawkins (2004)
   - Teoría del neocórtex como sistema predictivo

2. **A Thousand Brains** - Jeff Hawkins (2021)
   - Teoría de múltiples mapas corticales

3. **Consciousness Explained** - Daniel Dennett (1991)
   - Crítica al teatro cartesiano

4. **Parallel Distributed Processing** - Rumelhart & McClelland (1986)
   - Fundamentos del conexionismo

**Recursos Online:**

- Numenta Research: https://numenta.com/research
- Distill.pub: https://distill.pub (visualizaciones interactivas)
- HTM School: https://www.youtube.com/c/NumentaTheory (videos explicativos)

---

## 📄 README.md del Proyecto

```markdown
# 🧠 Connectionist Neural Automaton (CNA)

**Un Autómata Celular Neuronal para la Búsqueda de Inteligencia Artificial**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tuusuario/CNA_Project/blob/master/notebooks/01_Automata_Base.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tuusuario/CNA_Project/HEAD?labpath=notebooks)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Visión

Este proyecto implementa un **autómata celular neuronal** que unifica:

- **Kohonen** (Self-Organizing Maps, inhibición lateral tipo sombrero mexicano)
- **Hawkins** (Hierarchical Temporal Memory, predicción temporal)
- **Place/Grid Cells** (mapas espaciales emergentes, navegación)
- **Dennett** (consciencia distribuida, sin teatro cartesiano)
- **Neural Cellular Automata** (reglas aprendidas, no hardcoded)

**Objetivo:** Cerrar la brecha entre IA de lenguaje y robótica móvil, creando cerebros de bajo nivel escalables desde organismos simples (Aplysia, mosca) hasta sistemas complejos.

---

## ✨ Características

- ✅ **Autómata Celular Neuronal** con reglas emergentes (Hebbian learning)
- ✅ **Self-Organizing Maps** con inhibición lateral (Mexican Hat)
- ✅ **Memoria Temporal** (HTM) con predicción de secuencias
- ✅ **Place & Grid Cells** para navegación espacial
- ✅ **UI Interactiva** (ipycanvas) para dibujar y visualizar
- ✅ **Robot Simulado** con aprendizaje sensorimotor
- ✅ **Notebooks didácticos** en español con explicaciones paso a paso

---

## 🚀 Inicio Rápido

### Opción 1: Google Colab (Recomendado)

1. Click en el badge de Colab arriba
2. Runtime > Change runtime type > GPU
3. Ejecutar celdas secuencialmente

### Opción 2: Local

```bash
# Clonar repositorio
git clone https://github.com/tuusuario/CNA_Project.git
cd CNA_Project

# Crear entorno
conda env create -f environment.yml
conda activate cna

# O con pip
pip install -r requirements.txt

# Lanzar Jupyter
jupyter lab
```

---

## 📚 Notebooks

1. **[01_Automata_Base.ipynb](notebooks/01_Automata_Base.ipynb)**
   - Clases base: Neurona, Dendrita, Sinapsis
   - Autómata celular con regiones (ENTRADA/SALIDA/INTERNA)
   - Experimentos: Propagación de onda, reflejos simples

2. **[02_SOM_Kohonen.ipynb](notebooks/02_SOM_Kohonen.ipynb)**
   - Self-Organizing Maps
   - Función Mexican Hat (inhibición lateral)
   - Clustering de colores, mapas tonotópicos

3. **[03_HTM_Prediccion.ipynb](notebooks/03_HTM_Prediccion.ipynb)**
   - Sparse Distributed Representations (SDR)
   - Temporal Memory (secuencias)
   - Place & Grid Cells (navegación espacial)
   - Memory Replay

4. **[04_UI_Robotica.ipynb](notebooks/04_UI_Robotica.ipynb)**
   - UI interactiva con ipycanvas
   - Robot simulado navegando con CNA
   - Aprendizaje sensorimotor

---

## 🧪 Ejemplos de Uso

### Crear un CNA

```python
from src.cna import ConnessionistNeuralAutomaton

# Crear autómata 64x64
cna = ConnessionistNeuralAutomaton(64, 64, connect_radius=3)

# Activar región de entrada
import numpy as np
cna.set_input_region(np.random.rand(64) * 0.5)

# Ejecutar 100 pasos
for _ in range(100):
    cna.step()

# Visualizar
cna.visualize()
```

### Robot con CNA

```python
from src.robot import SimpleRobot, RobotBrainInterface

# Crear robot
robot = SimpleRobot(grid_size=(20, 20))
robot.set_goal((18, 18))

# Conectar con CNA
interface = RobotBrainInterface(robot, cna_size=(32, 32))

# Entrenar navegación
for episode in range(10):
    interface.train_episode(max_steps=100, visualize=True)
```

---

## 🔬 Fundamentos Científicos

Este proyecto se basa en investigaciones recientes:

- **Zebrafish whole-brain imaging** (2024): Mapeo completo de actividad neuronal
- **Predictive grid cells** (2024): Células que predicen posición futura
- **Neural Cellular Automata** (2020, Google): Autómatas con reglas aprendidas
- **HTM Theory** (Numenta): Memoria temporal jerárquica
- **Place & Grid Cells** (Nobel 2014): Mapas espaciales en hipocampo

Ver [PROYECTO_JUPYTER.md](PROYECTO_JUPYTER.md) para detalles completos.

---

## 🤝 Contribuir

¡Contribuciones son bienvenidas!

1. Fork el proyecto
2. Crea tu rama (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📜 Licencia

MIT License - Ver [LICENSE](LICENSE) para detalles

---

## 🙏 Agradecimientos

- **Jeff Hawkins** (Numenta) - Teoría HTM
- **Teuvo Kohonen** - Self-Organizing Maps
- **Daniel Dennett** - Consciencia distribuida
- **Eric Kandel** - Neurociencia de Aplysia
- **Google Research** - Neural Cellular Automata

---

## 📧 Contacto

Tu Nombre - [@tutwitter](https://twitter.com/tutwitter) - email@ejemplo.com

Proyecto Link: [https://github.com/tuusuario/CNA_Project](https://github.com/tuusuario/CNA_Project)

---

**⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!**
```

---

## 🎓 Conclusión

Has creado un **Connectionist Neural Automaton** completo que:

1. ✅ **Unifica teorías neurocientíficas** (Kohonen, Hawkins, Dennett)
2. ✅ **Escala desde sistemas simples** (Aplysia) a complejos (pez cebra)
3. ✅ **Integra percepción y acción** (robótica embodied)
4. ✅ **Aprende reglas emergentes** (no hardcoded)
5. ✅ **Es reproducible y educativo** (notebooks en español)

### Próximos Desafíos

1. **Conectar con transformers** para procesamiento simbólico de alto nivel
2. **Jerarquías profundas** (HTM multi-capa)
3. **Entornos 3D** (simuladores físicos)
4. **Hardware dedicado** (neuromorphic computing)
5. **Aplicaciones reales** (drones, robots móviles, prótesis)

### El Camino hacia la IA Completa

```
┌─────────────────────────────────────────────────────────┐
│                    ARQUITECTURA COMPLETA                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [NIVEL SIMBÓLICO]                                      │
│   Transformers, LLMs                                     │
│   • Razonamiento abstracto                              │
│   • Lenguaje natural                                     │
│            ↕ Embeddings ↕                               │
│  [NIVEL INTERMEDIO]                                      │
│   Hierarchical Temporal Memory                           │
│   • Patrones temporales                                  │
│   • Memoria episódica                                    │
│            ↕ Secuencias ↕                               │
│  [NIVEL BAJO - CNA]                                      │
│   Connectionist Neural Automaton                         │
│   • Mapas auto-organizados                              │
│   • Navegación espacial                                  │
│   • Control sensorimotor                                 │
│            ↕ Sensores ↕                                 │
│  [MUNDO FÍSICO]                                          │
│   Robot, Cuerpo, Entorno                                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

**¡Has dado el primer paso hacia la construcción de una IA verdaderamente embodied!** 🚀🧠🤖

---

*Documento creado por: [Tu Nombre]*  
*Fecha: Febrero 2026*  
*Versión: 1.0*
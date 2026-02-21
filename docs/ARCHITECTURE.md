# Arquitectura Técnica

Diseño técnico del sistema: stack, clases, API, protocolo y hosting.

Para la visión y filosofía del proyecto, ver [Visión](VISION.md).
Para la hoja de ruta, ver [Etapas](STAGES.md).
Para el modelo neuronal cercano al código, ver [Modelo Neuronal](../backend/core/README.md).

---

## 1. Stack Tecnológico

```
┌─────────────────────────────────────────────────────────┐
│                      FRONTEND                           │
│   Vite + React + TypeScript + HTML5 Canvas              │
│   Deploy: Vercel (gratis)                               │
└──────────────────────┬──────────────────────────────────┘
                       │ WebSocket (ws://)
                       │ frames en tiempo real
┌──────────────────────┴──────────────────────────────────┐
│                      BACKEND                            │
│   Python 3.11+ / FastAPI / uvicorn                      │
│   NumPy para operaciones matriciales                    │
│   Deploy: Render.com (gratis, 750h/mes)                 │
└─────────────────────────────────────────────────────────┘
```

### Justificación

| Componente | Elección | Por qué |
|------------|----------|---------|
| Backend framework | **FastAPI** | Async nativo, WebSocket, tipado, el más popular en Python 2025-2026 |
| Backend runtime | **uvicorn** | ASGI server estándar para FastAPI |
| Cómputo | **NumPy** | Operaciones matriciales vectorizadas, libera GIL |
| Frontend bundler | **Vite** | Build instantáneo, HMR, estándar actual |
| Frontend framework | **React 19 + TypeScript** | El más adoptado, componentes reutilizables |
| Renderizado | **HTML5 Canvas** | Directo, rápido, perfecto para grids de pixels |
| Comunicación | **WebSocket** | Bidireccional, baja latencia, ideal para streaming de frames |
| Hosting backend | **Render.com** | Único con free tier real para Python (750h/mes) |
| Hosting frontend | **Vercel** | Free tier generoso, deploy automático desde Git, óptimo para React |
| Tests backend | **pytest** | Estándar en Python, simple, potente |
| Tests frontend | **Vitest** | Nativo de Vite, compatible con Jest API |

---

## 2. Estructura del Proyecto

```
NeuroFlow/
├── backend/
│   ├── main.py                    # Entry point FastAPI
│   ├── requirements.txt           # Dependencias Python
│   ├── core/                      # Modelo neuronal (port de RedJavaScript)
│   │   ├── __init__.py
│   │   ├── sinapsis.py            # Conexión sináptica
│   │   ├── dendrita.py            # Rama dendrítica
│   │   ├── neurona.py             # Neurona + NeuronaEntrada
│   │   ├── red.py                 # Red neuronal (contenedor tonto)
│   │   ├── region.py              # Agrupación de neuronas (organización)
│   │   └── constructor.py         # Factory/builder de redes y regiones
│   ├── experiments/               # Experimentos (plug-in)
│   │   ├── __init__.py
│   │   ├── base.py                # Clase base Experiment
│   │   └── deamons_lab.py         # Deamons Lab (laboratorio de conexionados)
│   ├── api/
│   │   ├── __init__.py
│   │   ├── websocket.py           # Handler WebSocket
│   │   └── routes.py              # Endpoints REST
│   └── tests/
│       ├── conftest.py
│       ├── test_sinapsis.py
│       ├── test_dendrita.py
│       ├── test_neurona.py
│       ├── test_red.py             # Red NO sabe de regiones
│       ├── test_region.py          # Region es solo agrupación
│       ├── test_constructor.py     # Constructor arma Red + Regiones
│       ├── test_deamons_lab.py
│       └── test_red_tensor.py
│
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx               # Entry point React
│       ├── App.tsx                 # Layout principal
│       ├── components/
│       │   ├── PixelCanvas.tsx     # Renderizado de la grilla
│       │   ├── Sidebar.tsx         # Panel de experimentos
│       │   └── Controls.tsx        # Play/Pause/Step/Reset
│       ├── hooks/
│       │   └── useExperiment.ts    # WebSocket + estado del experimento
│       └── types/
│           └── index.ts           # Tipos compartidos
│
├── docs/
│   ├── ARCHITECTURE.md            # Este documento
│   ├── VISION.md                  # Filosofía, daemons, modelo de la mente
│   ├── STAGES.md                  # Hoja de ruta (5 etapas)
│   ├── REFERENCES.md              # Bibliografía completa
│   └── AUTHOR.md                  # Sobre el autor y dedicatoria
│
├── README.md                      # Punto de entrada, navegación
└── .gitignore
```

---

## 3. Modelo Neuronal (Core)

Port del modelo de RedJavaScript a Python, con una mejora arquitectónica clave:
**separación de responsabilidades entre procesamiento y organización**.

### 3.0 Principio de Diseño: Separación de Responsabilidades

En el proyecto original (RedJavaScript), la clase `Red` conocía las regiones
(ENTRADA, SALIDA, INTERNA) y decidía qué neuronas procesar. Esto acopla
organización con procesamiento.

En NeuroFlow separamos estas responsabilidades:

```
PROCESAMIENTO (no sabe de organización)     ORGANIZACIÓN (no sabe de procesamiento)
┌──────────────────────────────┐            ┌──────────────────────────────┐
│  Red                         │            │  Region                      │
│  Solo contiene neuronas.     │            │  Grupo nombrado de neuronas. │
│  Solo las procesa a todas.   │            │  Solo referencias, no dueña. │
│  No sabe qué es input/output.│            │  Útil para conectar, activar │
│                              │            │  y leer subconjuntos.        │
│  Sinapsis → Dendrita →       │            │                              │
│  Neurona                     │            │  Constructor                 │
│  (cada una sabe procesarse)  │            │  Crea neuronas, regiones,    │
│                              │            │  conectividad. Sabe de       │
└──────────────────────────────┘            │  topología y reglas.         │
                                            │                              │
                                            │  Experimento                 │
                                            │  Orquesta: qué es entrada,  │
                                            │  qué es salida, cómo se     │
                                            │  alimenta, cómo se lee.     │
                                            └──────────────────────────────┘
```

**¿Por qué?** (respaldado por la literatura)
- **Modular Deep Learning** (arXiv 2023): Separar computación de routing/organización
  permite módulos autónomos, transferencia positiva y generalización sistemática.
- **PyTorch nn.Module**: El contenedor es tonto, solo hace `forward()`.
  No sabe si es "capa de entrada" o "capa de salida". Eso lo decide quien compone.
- **Martin Fowler (Domain Model + Factory/Builder)**: Factory para crear elementos
  livianos (neuronas), Builder para configuraciones complejas (regiones + conectividad).
- **Single Responsibility Principle**: La Red procesa. El Constructor organiza.
  El Experimento orquesta.

### 3.1 Diagrama de Clases

```
═══════════════════════════════════════════════════════════════════
  CAPA DE PROCESAMIENTO (core/) — No sabe de organización
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────┐
│                        Red                          │
│─────────────────────────────────────────────────────│
│  neuronas: dict[str, Neurona]                       │
│─────────────────────────────────────────────────────│
│  procesar()     → procesa TODAS las neuronas        │
│  get_grid(w, h) → retorna matriz de valores         │
│  get_neurona(id) → retorna neurona por id           │
│─────────────────────────────────────────────────────│
│  NO tiene regiones.                                 │
│  NO sabe qué es input ni output.                    │
│  Solo itera y procesa lo que le dieron.             │
└────────────┬────────────────────────────────────────┘
             │ contiene N
             ▼
┌─────────────────────────────────────────────────────┐
│                     Neurona                         │
│─────────────────────────────────────────────────────│
│  id: str                                            │
│  valor: float {0, 1}                                │
│  tension_superficial: float [-1, 1]                 │
│  dendritas: list[Dendrita]                          │
│  umbral: float                                      │
│─────────────────────────────────────────────────────│
│  procesar()   → fuzzy OR de dendritas               │
│  activar()    → umbral sobre tensión                │
│  entrenar()   → propaga entrenamiento               │
│─────────────────────────────────────────────────────│
│                                                     │
│  ┌────────────────────────────────────────────┐     │
│  │         NeuronaEntrada (hereda)            │     │
│  │  Sin dendritas.                            │     │
│  │  procesar() → no-op                        │     │
│  │  activar()  → no-op                        │     │
│  │  activar_external(valor) → setea valor     │     │
│  │                                            │     │
│  │  La Red la procesa igual que las demás,    │     │
│  │  pero ella internamente no hace nada.      │     │
│  │  La Red NO necesita saber que es especial. │     │
│  └────────────────────────────────────────────┘     │
└────────────┬────────────────────────────────────────┘
             │ contiene M
             ▼
┌─────────────────────────────────────────────────────┐
│                    Dendrita                          │
│─────────────────────────────────────────────────────│
│  peso: float [-1, 1]    ← PUEDE SER NEGATIVO       │
│  valor: float                                       │
│  sinapsis: list[Sinapsis]                           │
│─────────────────────────────────────────────────────│
│  procesar()   → avg(sinapsis) * peso  (fuzzy AND)   │
│  entrenar()   → propaga a sinapsis                  │
│─────────────────────────────────────────────────────│
│  Nota: puede tener UNA sola sinapsis si se requiere │
└────────────┬────────────────────────────────────────┘
             │ contiene K
             ▼
┌─────────────────────────────────────────────────────┐
│                    Sinapsis                          │
│─────────────────────────────────────────────────────│
│  peso: float [0, 1]     ← SIEMPRE POSITIVO         │
│  valor: float                                       │
│  neurona_entrante: Neurona (referencia al axón)     │
│─────────────────────────────────────────────────────│
│  procesar()   → 1 - |peso - entrada|               │
│  entrenar()   → Hebbian: peso += (entrada - peso)*η│
└─────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════
  CAPA DE ORGANIZACIÓN (core/) — No sabe de procesamiento
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────┐
│                    Region                           │
│─────────────────────────────────────────────────────│
│  nombre: str                                        │
│  neuronas: dict[str, Neurona]  ← referencias        │
│─────────────────────────────────────────────────────│
│  agregar(neurona)                                   │
│  ids() → lista de ids                               │
│  valores() → lista de valores                       │
│─────────────────────────────────────────────────────│
│  NO es dueña de las neuronas (solo referencia).     │
│  La Red no sabe que existen regiones.               │
│  Es una herramienta para el Constructor y el        │
│  Experimento, no para la Red.                       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                  Constructor                        │
│─────────────────────────────────────────────────────│
│  Crea neuronas, las agrupa en regiones,             │
│  construye la conectividad (dendritas, sinapsis).   │
│─────────────────────────────────────────────────────│
│  crear_grilla(w, h)  → Red + dict de regiones       │
│  crear_region(nombre, neuronas) → Region            │
│  conectar(origen, destino, mascara_relativa)        │
│  aplicar_regla_wolfram(regla, neuronas, vecinos)    │
│─────────────────────────────────────────────────────│
│  Conoce de topología y patrones de conexión.        │
│  Es el ÚNICO que sabe cómo cablear la red.          │
│  Una vez construida, la Red funciona sola.          │
└─────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════
  CAPA DE ORQUESTACIÓN (experiments/) — Usa todo lo anterior
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────┐
│              Experimento (base)                     │
│─────────────────────────────────────────────────────│
│  red: Red                                           │
│  regiones: dict[str, Region]                        │
│─────────────────────────────────────────────────────│
│  setup(config)  → usa Constructor para armar todo   │
│  step()         → red.procesar() + retorna frame    │
│  click(x, y)    → busca neurona en región entrada   │
│  reset()        → reinicia                          │
│  get_frame()    → red.get_grid()                    │
│─────────────────────────────────────────────────────│
│  SABE qué región es entrada y cuál es salida.       │
│  SABE cómo alimentar la red y leer resultados.      │
│  La Red no sabe nada de esto.                       │
└─────────────────────────────────────────────────────┘
```

### 3.2 Flujo de Responsabilidades

```
Experimento (orquesta)
  │
  │  1. setup: pide al Constructor que arme la red
  │
  ▼
Constructor (organiza)
  │
  │  2. Crea neuronas (Neurona y NeuronaEntrada)
  │  3. Las agrupa en regiones
  │  4. Conecta dendritas y sinapsis según la regla
  │  5. Entrega: Red + dict de Regiones
  │
  ▼
Red (procesa) ◄── Regiones (referencias)
  │
  │  6. Experimento llama red.procesar()
  │  7. Red itera TODAS las neuronas:
  │     - NeuronaEntrada.procesar() → no-op (ella ya tiene su valor)
  │     - Neurona.procesar() → evalúa dendritas → sinapsis
  │  8. Red itera TODAS las neuronas:
  │     - NeuronaEntrada.activar() → no-op
  │     - Neurona.activar() → umbral sobre tensión
  │
  ▼
Experimento lee red.get_grid() → frame → WebSocket → Frontend
```

### 3.3 Lógica de Procesamiento

```
SINAPSIS:   valor = 1 - |peso - neurona_entrante.valor|
            Si peso=1 y entrada=1 → 1 (match perfecto)
            Si peso=0 y entrada=0 → 1 (match perfecto)
            Si peso=1 y entrada=0 → 0 (no match)
            Si peso=0 y entrada=1 → 0 (no match)

DENDRITA:   valor = promedio(sinapsis.procesar()) × peso_dendrita
            Fuzzy AND: todas las sinapsis deben matchear
            peso_dendrita puede ser negativo → inhibición

NEURONA:    max_dendrita = max(dendritas.valor)
            min_dendrita = min(dendritas.valor)   (negativas)
            tension = max + min                   (competencia)
            Si tension > umbral → valor = 1
            Si no → valor = 0
            Fuzzy OR: cualquier dendrita positiva puede activar
            Pero dendritas negativas pueden inhibir

NEURONA_ENTRADA:
            procesar() → no-op (no tiene dendritas)
            activar()  → no-op (su valor ya fue seteado)
            Solo cambia vía activar_external(valor) desde el Experimento
```

### 3.4 Reglas de Peso

```
SINAPSIS.peso ∈ [0, 1]     ← Siempre positivo o cero
                               Representa "reconocimiento de patrón"
                               peso≈1 reconoce entrada=1
                               peso≈0 reconoce entrada=0

DENDRITA.peso ∈ [-1, 1]    ← Puede ser negativo
                               peso > 0: dendrita excitatoria
                               peso < 0: dendrita inhibitoria
                               Permite implementar NOT/inhibición
```

### 3.5 Analogía con PyTorch

```
PyTorch                          NeuroFlow
─────────────────────────────    ─────────────────────────────
nn.Module (forward)          →   Red (procesar)
  No sabe si es input/output       No sabe de regiones
  Solo computa                     Solo itera neuronas

nn.Sequential / Model        →   Constructor
  Compone módulos en orden         Arma la Red con regiones
  Define la topología              Define conectividad

Training loop                →   Experimento
  Alimenta datos                   Alimenta entradas
  Lee salidas                      Lee la grilla
  Orquesta todo                    Orquesta todo
```

### 3.6 Máscaras de Conexionado (masks.py)

Las máscaras definen la topología de conexión de cada neurona: qué vecinos son
excitatorios y cuáles inhibitorios. Se configuran como presets en `backend/core/masks.py`
y se cargan dinámicamente desde la API.

```
         Excitación (E)          Gap (G)           Inhibición (I)
        ┌───────────┐      ┌──────────────┐      ┌───────────────┐
        │  Moore r=n │      │  sin conexión │      │  anillo/corona │
        │  (vecinos  │      │  (silencio)   │      │  8 dendritas   │
        │  cercanos) │      │              │      │  sectorizada   │
        └───────────┘      └──────────────┘      └───────────────┘
```

#### Nomenclatura Deamon

Las máscaras tipo Deamon usan la convención `E G I [DE DI]`:

```
E<n>   Radio excitatorio: Moore r=n
G<n>   Gap: n anillos de silencio entre excitación e inhibición
I<n>   Radio inhibitorio: n anillos de corona
DE<n>  Densidad excitatoria: fracción 1/n de sinapsis (random, seed fija)
DI<n>  Densidad inhibitoria: fracción 1/n de sinapsis (random, seed fija)
```

Ejemplo: `E2 G3 I3 DE1 DI1.5` → Moore r=2 completa, 3 anillos de gap,
3 anillos inhibitorios con ~67% de densidad.

DE/DI omitidos implican densidad 1 (completa). La densidad usa `_random_sparse()`
con seed fija para que la máscara sea determinista entre ejecuciones pero con
distribución espacial aleatoria (a diferencia de `_sparse_ring` que usa patrones
tipo checkerboard).

#### Helpers de Generación

| Helper | Descripción |
|--------|-------------|
| `_moore(r)` | Vecindad Moore: Chebyshev dist ≤ r |
| `_ring(r_in, r_out)` | Anillo: Chebyshev dist ∈ [r_in, r_out] |
| `_von_neumann(r)` | Vecindad Von Neumann: Manhattan dist ≤ r |
| `_sparse_ring(r_in, r_out, step)` | Anillo sparse determinista (checkerboard) |
| `_random_sparse(offsets, density, seed)` | Submuestreo aleatorio con seed fija |
| `_make_inhibitory(offsets, peso, n)` | Particiona offsets en n sectores inhibitorios |
| `_partition(offsets, n)` | Divide offsets en n sectores angulares |

---

## 4. Experimento 0: Autómata Elemental (Von Neumann)

### 4.1 Concepto

Un autómata celular elemental (1D, reglas de Wolfram) implementado
íntegramente con el modelo neuronal. La grilla 2D muestra el diagrama
espacio-temporal: cada fila es una generación del autómata.

```
      Columnas (espacio, 50 celdas)
      ←─────────────────────────────→

  ↑   ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐   Fila 0: SALIDA (última generación)
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤   Filas internas: INTERNA
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤   (procesadas bottom-up)
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
Flujo ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │   └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘   Fila 49: ENTRADA (condición inicial)
                                        ← El usuario hace click aquí
```

### 4.2 Conexionado Neural para Rule 111

Rule 111 en binario: `01101111`

| Patrón (izq, centro, der) | Decimal | Resultado |
|---------------------------|---------|-----------|
| 1, 1, 1                  | 7       | **0**     |
| 1, 1, 0                  | 6       | **1**     |
| 1, 0, 1                  | 5       | **1**     |
| 1, 0, 0                  | 4       | **0**     |
| 0, 1, 1                  | 3       | **1**     |
| 0, 1, 0                  | 2       | **1**     |
| 0, 0, 1                  | 1       | **1**     |
| 0, 0, 0                  | 0       | **1**     |

Cada neurona interna en posición (x, y) se conecta con las 3 neuronas
de la fila de abajo: (x-1, y+1), (x, y+1), (x+1, y+1).

```
Fila y:     [  ?  ]  ← neurona a calcular
              / | \
Fila y+1: [izq][cen][der]  ← 3 entradas
```

**Implementación con 6 dendritas** (una por cada patrón que produce 1):

```
Dendrita 1 → patrón 110: sinapsis pesos [1, 1, 0] → peso_dendrita = +1
Dendrita 2 → patrón 101: sinapsis pesos [1, 0, 1] → peso_dendrita = +1
Dendrita 3 → patrón 011: sinapsis pesos [0, 1, 1] → peso_dendrita = +1
Dendrita 4 → patrón 010: sinapsis pesos [0, 1, 0] → peso_dendrita = +1
Dendrita 5 → patrón 001: sinapsis pesos [0, 0, 1] → peso_dendrita = +1
Dendrita 6 → patrón 000: sinapsis pesos [0, 0, 0] → peso_dendrita = +1
```

Cuando el patrón de entrada es, por ejemplo, `1 1 0`:
- Dendrita 1 (110): sinapsis → [1-|1-1|, 1-|1-1|, 1-|0-0|] = [1, 1, 1] → avg=1.0 ✓
- Dendrita 2 (101): sinapsis → [1-|1-1|, 1-|0-1|, 1-|1-0|] = [1, 0, 0] → avg=0.33 ✗
- ...solo la dendrita 1 da valor alto → neurona se activa → **1** ✓

### 4.3 Procesamiento por Frames

```
Frame 0:  Solo fila 49 visible (ENTRADA, click del usuario)
Frame 1:  Se procesa fila 48 (lee fila 49)
Frame 2:  Se procesa fila 47 (lee fila 48)
...
Frame 49: Se procesa fila 0 (SALIDA)

Total: 49 frames para llenar toda la grilla
```

### 4.4 Reglas Adicionales Planificadas

| Regla | Tipo | Descripción |
|-------|------|-------------|
| Rule 111 | Determinista | Primer test, muchos 1s |
| Rule 30 | Caótica | Triángulos de Sierpinski, caos |
| Rule 90 | Fractal | Triángulo de Sierpinski perfecto |
| Rule 110 | Turing-completa | La más interesante teóricamente |

Cada regla solo requiere reconfigurar qué dendritas tiene cada neurona.
El modelo neuronal (Sinapsis, Dendrita, Neurona, Red) no cambia.

---

## 5. API y Comunicación

### 5.1 REST Endpoints

```
GET  /api/experiments
     → [{ id: "deamons_lab", name: "Deamons Lab", masks: [...] }]

GET  /api/experiments/:id
     → { id, name, description, default_config: { width: 30, height: 30, mask: "simple" } }

GET  /api/health
     → { status: "ok", version: "0.1.0" }
```

### 5.2 WebSocket Protocol

```
Conexión: ws://host/ws/experiment

─── Cliente → Servidor ───────────────────────────────────

{ "action": "start",
  "experiment": "deamons_lab",
  "config": { "width": 30, "height": 30, "mask": "simple" } }

{ "action": "click", "x": 25, "y": 49 }    // Activar neurona

{ "action": "step" }                         // Avanzar 1 frame
{ "action": "play" }                         // Animación continua
{ "action": "pause" }                        // Pausar
{ "action": "reset" }                        // Reiniciar

─── Servidor → Cliente ───────────────────────────────────

{ "type": "frame",
  "generation": 5,
  "grid": [[0,1,0,...], [1,1,0,...], ...],   // Matriz 50x50
  "stats": {
    "active_cells": 123,
    "processed_rows": 5,
    "total_rows": 50
  }
}

{ "type": "status",
  "state": "running" | "paused" | "ready" | "complete" }

{ "type": "error",
  "message": "..." }
```

### 5.3 Flujo de Datos

```
┌──────────────────────┐          ┌──────────────────────────────────┐
│      FRONTEND        │          │            BACKEND               │
│                      │          │                                  │
│  ┌────────────────┐  │  start   │  ┌────────────────────────────┐  │
│  │   Sidebar      │──┼─────────►│  │   Experimento (orquesta)  │  │
│  │  (experiments) │  │          │  │                            │  │
│  └────────────────┘  │          │  │  setup:                    │  │
│                      │          │  │   Constructor → Red        │  │
│  ┌────────────────┐  │  click   │  │               + Regiones   │  │
│  │  PixelCanvas   │──┼─────────►│  │                            │  │
│  │  (HTML5 Canvas)│  │          │  │  click(x,y):               │  │
│  │  50×50 pixels  │  │          │  │   region_entrada           │  │
│  └────────▲───────┘  │          │  │     .get(x,y)              │  │
│           │          │          │  │     .activar_external(1)   │  │
│  ┌────────┴───────┐  │  frame   │  │                            │  │
│  │  useExperiment │◄─┼──────────│  │  step:                     │  │
│  │  (WebSocket)   │  │          │  │   red.procesar()  ← tonta  │  │
│  └────────────────┘  │          │  │     Neurona.procesar()     │  │
│                      │          │  │       Dendrita.procesar()  │  │
│  ┌────────────────┐  │          │  │         Sinapsis.procesar()│  │
│  │  Controls      │──┼─────────►│  │   red.get_grid() → frame  │  │
│  │  Play/Pause    │  │  step    │  │                            │  │
│  └────────────────┘  │          │  └────────────────────────────┘  │
└──────────────────────┘          └──────────────────────────────────┘
```

---

## 6. Frontend: Diseño de UI

```
┌─────────────────────────────────────────────────────────────┐
│  NeuroFlow                                          v0.1.0  │
├──────────────┬──────────────────────────────────────────────┤
│              │                                              │
│ EXPERIMENTOS │         GRILLA DE NEURONAS                   │
│              │                                              │
│ ● Exp 0:    │    ┌──────────────────────────────┐           │
│   Von Neumann│    │  ■ □ □ ■ □ ■ ■ □ □ ■ ...  │  SALIDA   │
│              │    │  □ ■ □ □ ■ □ □ ■ □ □ ...  │           │
│   Regla:     │    │  ■ ■ □ ■ ■ ■ □ □ ■ □ ...  │           │
│   [111 ▼]    │    │  □ □ ■ □ □ □ ■ □ □ ■ ...  │  INTERNA  │
│              │    │  ...                        │           │
│   Tamaño:    │    │  □ □ □ □ □ ■ □ □ □ □ ...  │           │
│   50 × 50    │    │  □ □ □ □ □ □ □ □ □ □ ...  │  ENTRADA  │
│              │    └──────────────────────────────┘           │
│   Velocidad: │    ← Click para activar neuronas →           │
│   ████░░ 7fps│                                              │
│              │   ┌──────────────────────────────────┐       │
│ ○ Exp 1:    │   │  ▶ Play  ⏸ Pause  ⏭ Step  ↺ Reset │    │
│   Conway     │   └──────────────────────────────────┘       │
│   (próximo)  │                                              │
│              │   Gen: 23/50  │  Celdas activas: 147         │
│              │                                              │
├──────────────┴──────────────────────────────────────────────┤
│  ⬛ = neurona activa (valor=1)   □ = inactiva (valor=0)     │
│  🔵 = ENTRADA   🔴 = SALIDA   ⬜ = INTERNA                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Hosting Gratuito

### 7.1 Backend → Render.com

- **Plan**: Free tier (750 horas/mes)
- **Limitación**: Spin-down tras 15min de inactividad (~1min cold start)
- **Deploy**: Desde Git, auto-build con `requirements.txt`
- **Runtime**: Python 3.11, uvicorn

```yaml
# render.yaml (Blueprint)
services:
  - type: web
    name: neuroflow-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

### 7.2 Frontend → Vercel

- **Plan**: Hobby (gratis)
- **Build**: Vite produce archivos estáticos
- **Deploy**: Desde Git, auto-detect Vite

```json
// vercel.json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite"
}
```

### 7.3 CORS

El backend debe permitir requests del frontend:

```python
# main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://neuroflow.vercel.app", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 8. Plan de Implementación

### Fase 0: Walking Skeleton (este sprint)

```
1. [Tests]  → test_sinapsis.py, test_dendrita.py, test_neurona.py, test_red.py
2. [Core]   → sinapsis.py, dendrita.py, neurona.py, red.py, constructor.py
3. [Tests]  → test_deamons_lab.py, test_red_tensor.py
4. [API]    → main.py con WebSocket + endpoint de experimentos
5. [UI]     → React app con Canvas, sidebar, controles
6. [Exp]    → Deamons Lab (todos los conexionados + Wolfram) end-to-end
7. [Deploy] → Backend en Render, Frontend en Vercel
```

### Fase 1: Más Reglas + Conway

```
8.  Rule 30, 90, 110 (solo reconfiguración de dendritas)
9.  Experimento 1: Game of Life (Conway) - vecindad Moore (8 vecinos)
10. UI: selector de experimentos dinámico
```

### Fase 2: Aprendizaje Emergente

```
11. Activar entrenamiento Hebbiano
12. Poda sináptica
13. Visualización de pesos en tiempo real
```

### Fase 3: Deamons + HTM

```
14. Mapas auto-organizados
15. Memoria temporal jerárquica
16. Regiones funcionales (DOLOR)
```

---

## 9. Decisiones Técnicas Clave

### ¿Por qué no Jupyter Notebooks?

- No son desplegables como aplicación web
- Requieren instalación local
- La visualización interactiva es limitada
- No escalan a múltiples usuarios

### ¿Por qué separar frontend y backend?

- El cómputo neural puede ser pesado → backend dedicado
- La UI debe ser responsive → no bloquear con cómputo
- Permite escalar independientemente
- Permite usar GPU en backend sin afectar UI

### ¿Por qué WebSocket y no polling?

- El autómata produce ~10-30 frames/segundo
- Polling generaría demasiados HTTP requests
- WebSocket permite streaming bidireccional continuo
- El cliente puede enviar clicks sin latencia extra

### ¿Por qué React y no Svelte/Vue?

- React es el más adoptado y documentado
- Para un Canvas con sidebar, React es suficientemente simple
- Ecosistema más grande para futuras necesidades
- TypeScript support maduro

### ¿Por qué NumPy para el cómputo?

- Operaciones vectorizadas son ~100x más rápidas que loops Python
- Para 50×50 = 2500 neuronas, es instantáneo
- Escala bien hasta ~1000×1000 sin GPU
- Familiar para científicos e ingenieros

---

← Volver al [README](../README.md)

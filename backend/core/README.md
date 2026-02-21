# Modelo Neuronal — `backend/core/`

El corazón de NeuroFlow. Aquí vive el modelo conexionista: las piezas
fundamentales que procesan información sin controlador central.

---

## En una línea

> Sinapsis reconocen patrones → Dendritas los combinan (excitan o inhiben)
> → Neuronas compiten y se activan → la Red las procesa a todas sin saber
> nada de organización.

---

## Piezas fundamentales

```
Sinapsis (peso ≥ 0)  →  Dendrita (peso ∈ [-1,1])  →  Neurona (activa/inactiva)
   reconoce patrón         AND fuzzy + inhibición         OR fuzzy competitivo
```

### Sinapsis (`sinapsis.py`)

La unidad mínima de conexión. Conecta una neurona entrante con una
dendrita.

- **Peso**: siempre en `[0, 1]` — representa reconocimiento de patrón
- **Procesamiento**: `valor = 1 - |peso - entrada|`
  - `peso≈1` reconoce `entrada=1` (match perfecto → 1)
  - `peso≈0` reconoce `entrada=0` (match perfecto → 1)
  - Mismatch → valor bajo
- **Entrenamiento**: Hebbiano — `peso += (entrada - peso) × η`

### Dendrita (`dendrita.py`)

Rama de entrada que agrupa sinapsis y las combina.

- **Peso**: en `[-1, 1]` — puede ser negativo (inhibición)
- **Procesamiento**: `valor = promedio(sinapsis) × peso_dendrita` (fuzzy AND)
- Peso positivo → dendrita **excitatoria**
- Peso negativo → dendrita **inhibitoria**
- Puede tener una sola sinapsis si se requiere

### Neurona (`neurona.py`)

La unidad de decisión. Contiene dendritas y resuelve la competencia.

- **Valor**: `{0, 1}` — activa o inactiva
- **Tensión superficial**: `[-1, 1]` — acumulación antes de umbral
- **Procesamiento**:
  1. `max_dendrita = max(dendritas.valor)` (excitación)
  2. `min_dendrita = min(dendritas.valor)` (inhibición, si negativas)
  3. `tensión = max + min` (competencia)
  4. Si `tensión > umbral` → activa (1), si no → inactiva (0)
- Fuzzy OR: cualquier dendrita positiva puede activar
- Pero dendritas negativas pueden inhibir

**NeuronaEntrada** (hereda de Neurona): sin dendritas, su valor se setea
externamente. La Red la procesa igual pero ella no hace nada internamente.

### Red (`red.py`)

Contenedor "tonto" — no sabe de organización, solo procesa.

- Contiene un diccionario de neuronas
- `procesar()`: itera todas y las procesa (incluye NeuronaEntrada, que no-op)
- `get_grid(w, h)`: retorna la matriz de valores para visualización
- **No sabe** qué es entrada, salida, región ni experimento

---

## Organización (sin procesamiento)

### Region (`region.py`)

Agrupación nombrada de neuronas — solo referencias, no dueña.
La Red no sabe que existen regiones.

### Constructor (`constructor.py`)

Factory/Builder que arma la red: crea neuronas, las agrupa en regiones,
construye conectividad (dendritas, sinapsis) según reglas.

**Es el único que sabe cómo cablear la red.** Una vez construida, la Red
funciona sola.

### ConstructorTensor (`constructor_tensor.py`)

Versión tensorizada del constructor para cómputo matricial eficiente.

### RedTensor (`red_tensor.py`)

Versión tensorizada de la Red que opera con matrices NumPy en lugar de
objetos Python individuales.

---

## Máscaras de conexionado (`masks.py`)

Las máscaras definen la topología: qué vecinos son excitatorios, cuáles
inhibitorios, con qué densidad.

### Nomenclatura Daemon: `E G I [DE DI]`

| Parámetro | Significado | Ejemplo |
|-----------|-------------|---------|
| **E***n* | Radio excitatorio (Moore r=*n*) | E3 = Moore r=3 (48 vecinos) |
| **G***n* | Gap: anillos de silencio | G12 = 12 anillos sin conexión |
| **I***n* | Radio inhibitorio (corona) | I3 = 3 anillos inhibitorios |
| **DE***n* | Densidad excitatoria (1/*n*) | DE1 = completa, DE3 = ~33% |
| **DI***n* | Densidad inhibitoria (1/*n*) | DI1 = completa, DI1.5 = ~67% |

```
      Excitación (E)         Gap (G)          Inhibición (I)
    ┌─────────────┐    ┌──────────────┐    ┌───────────────┐
    │  Moore r=n  │    │  sin conexión │    │  anillo/corona │
    │  (vecinos   │    │  (silencio)   │    │  sectorizada   │
    │  cercanos)  │    │              │    │  (8 dendritas)  │
    └─────────────┘    └──────────────┘    └───────────────┘
```

**Ejemplo**: `E2 G3 I3 DE1 DI1.5`
- Excitación: Moore r=2 completa (24 vecinos)
- Gap: r=3–5 (3 anillos de silencio)
- Inhibición: r=6–8 con densidad 1/1.5 ≈ 67%

### Inspiración biológica

Esta estructura replica el **sombrero mexicano** (Mexican hat) observado
en el córtex visual por Hubel & Wiesel (1962) y formalizado por Kohonen
en los SOMs: excitación local rodeada de inhibición lateral — el patrón
fundamental que permite la auto-organización topográfica en el cerebro.

---

## Principio de diseño: separación de responsabilidades

```
PROCESAMIENTO                          ORGANIZACIÓN
(no sabe de organización)              (no sabe de procesamiento)

  Red                                    Region
  Solo procesa neuronas.                 Solo agrupa neuronas.
  No sabe de regiones.                   No las procesa.

                                         Constructor
                                         Crea y cablea.

                                         Experimento
                                         Orquesta todo.
```

Ver [Arquitectura Técnica](../../docs/ARCHITECTURE.md) para el diseño
completo con diagramas.

---

← Volver al [README](../../README.md)

# Hoja de Ruta

Las etapas siguen un orden lógico: cada una construye sobre la anterior.
El objetivo final es un agente motor inteligente que opere sin lenguaje,
emulando criaturas simples como *Aplysia* o el pez cebra.

---

## Resumen

```
Etapa 1   Etapa 2       Etapa 3              Etapa 4      Etapa 5
Daemon  → SOM Dinámico → Motor/Nociceptor  → Tuning     → Agentes Motores
(✓)       (próximo)      (teórico)           (genéticos)   (simulación)
```

---

## Etapa 1: Encontrar el Daemon ✓

**Estado:** Prácticamente cubierta.

**Objetivo:** Descubrir si una red puramente conexionista, con reglas
locales y sin procesamiento centralizado, puede producir unidades
estables de activación — los *daemons*.

**Qué se logró:**

| Logro | Descripción |
|-------|-------------|
| Movimiento | Los daemons se mueven: arriba, abajo, izquierda, derecha |
| Estabilidad | No se disipan; mantienen su forma |
| Resistencia al ruido | La señal se impone sobre el ruido |
| Exclusión competitiva | Los daemons compiten y se excluyen mutuamente |
| Balance natural | ~50% de neuronas activas en todo momento |
| Burbujas de activación | Al apagar una, otra se enciende; equilibrio dinámico |
| Convergencia | Al manipular externamente, el sistema converge a nuevo estado |
| Múltiples resoluciones | Diferentes conectados almacenan información a distintas escalas |

**Experimento:** *Deamons Lab* — laboratorio de conexionados con
presets tipo `E G I DE DI` (ver [Modelo Neuronal](../backend/core/README.md)).

**Hallazgo inesperado:** Los daemons se comportan como notas musicales
respecto a la exclusión competitiva (ver [Visión](VISION.md#paralelo-con-las-notas-musicales)).

**Siguiente:** Se retomará en la Etapa 4 (Tuning) con algoritmos genéticos.

---

## Etapa 2: SOM Dinámico

**Estado:** Próximo.

**Objetivo:** Implementar un mapa auto-organizativo (Self-Organizing Map)
usando el modelo conexionista de NeuroFlow y observar si el sistema
replica la capacidad de organización topográfica que Kohonen describió.

**Qué se busca:**

- Ver cómo el sistema se comporta de forma similar a un SOM clásico
- Mejorar cómo se almacenan y organizan las imágenes
- Observar la dinámica del SOM tanto para:
  - Conectados estables (daemons fijos)
  - Conectados con movimiento (sin perder capacidad de clustering)
- Implementar el **entrenamiento** dentro de esta etapa

**Experimento:** *Dynamic SOM* — nuevo experimento en la sidebar.

**Inspiración:** Kohonen (1990), Hubel & Wiesel (1981), la relación
entre SOMs y las capas de redes convolucionales (Deep Dream).

---

## Etapa 3: Motor y Nociceptor

**Estado:** Planificado.

**Objetivo:** Explorar dos conceptos fundamentales antes de pasar a
agentes motores completos:

### Nociceptor

El nociceptor es el receptor del dolor. En NeuroFlow, modelar la
**inhibición nociceptiva** permite entender cómo un sistema distribuido
puede señalar "peligro" sin un procesador central — análogo a la
teoría de compuerta del dolor (gate control theory) en el asta dorsal
de la médula espinal.

### Motor

Cómo interpretar las **salidas** de un sistema conexionista como
señales motoras. No se busca resolver un problema concreto en esta
etapa, sino:

- Desarrollar modelos teóricos abstractos
- Observar qué dinámicas emergen
- Definir qué métricas se pueden obtener

**Experimento:** *Motor y Nociceptor* — modelos teóricos y abstractos.

Cualquier métrica obtenida aquí alimentará la Etapa 4 (Tuning).

---

## Etapa 4: Tuning

**Estado:** Planificado.

**Objetivo:** Optimizar los conectados y parámetros del sistema usando
técnicas sistemáticas de búsqueda y selección.

**Herramientas:**

| Herramienta | Para qué |
|-------------|----------|
| **Algoritmos genéticos** | Variar conectados (máscaras) y seleccionar por métricas |
| **scikit-learn** | Comparar propuestas de red, seleccionar por mérito (GridSearchCV, pipelines) |

**Parámetros a optimizar:**

- Cantidad de neuronas, sinapsis y dendritas
- Zona excitatoria, gap e inhibitoria (`E G I`)
- Densidad de cada zona (`DE DI`)
- Ubicación de sinapsis respecto al axón
- Posición del axón respecto al centro de masa de conexiones

**Métricas a pulir (deben estar maduras para esta etapa):**

- Cantidad de daemons
- Nivel de ruido
- Velocidad de formación desde ruido
- Estabilidad de cada candidato

**Se retoma:** El laboratorio de Daemons con estas herramientas de
optimización.

---

## Etapa 5: Agentes Motores

**Estado:** Planificado (horizonte largo).

**Objetivo:** Llevar el modelo conexionista a un mundo simulado donde
un agente se mueva de forma inteligente sin lenguaje.

**Plan:**

1. Simular un mundo virtual con un agente simple (esfera u objeto mínimo)
2. Publicar modelos que emulen seres vivos sin lenguaje:
   - ***Aplysia californica***: babosa marina estudiada por Eric Kandel;
     sistema nervioso con ~20.000 neuronas, permitió descubrir los
     mecanismos sinápticos del aprendizaje y la memoria
   - **Pez cebra** (*Danio rerio*): sistema nervioso transparente y
     relativamente simple; circuitos motores espinales bien
     caracterizados para locomoción

**Inspiración científica:**

- Kandel, E. R. (2001). *The Molecular Biology of Memory Storage: A
  Dialogue Between Genes and Synapses*. Nobel Lecture.
- Modelos computacionales de circuitos locomotores espinales en pez
  cebra (eLife, 2021; Nature Neuroscience, 2023).

---

## Líneas futuras (sin etapa asignada)

| Línea | Descripción |
|-------|-------------|
| Generación musical | Modelo neuronal que sintetice música a partir de la dinámica de exclusión competitiva |
| Generación de imágenes | Explorar la conexión entre SOMs, Deep Dream y el modelo conexionista |
| Modelos 3D | Extender el tejido 2D a volúmenes neuronales 3D |
| Lenguaje | Abordar modelado de lenguaje desde el enfoque conexionista |

---

← Volver al [README](../README.md)

# Experimentos — `backend/experiments/`

Los experimentos orquestan el modelo neuronal: deciden qué es entrada,
qué es salida, cómo se alimenta la red y cómo se leen los resultados.

---

## En una línea

> Cada experimento usa el Constructor para armar una Red con Regiones,
> y luego la procesa frame a frame vía WebSocket.

---

## Estructura

Todos los experimentos heredan de `base.py`:

```
Experimento (base)
├── setup(config)  → usa Constructor para armar Red + Regiones
├── step()         → red.procesar() + retorna frame
├── click(x, y)    → manipula neurona en región de entrada
├── reset()        → reinicia
└── get_frame()    → red.get_grid()
```

---

## Experimentos actuales

### Deamons Lab (`deamons_lab.py`)

**Etapa:** 1 — Encontrar el Daemon (✓ prácticamente cubierta)

Laboratorio de conexionados donde se prueban distintas máscaras Daemon
(`E G I DE DI`) para observar la dinámica de daemons: formación,
estabilidad, movimiento, exclusión competitiva y balance natural.

**Configuración:**

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `width` | Ancho de la grilla | 30 |
| `height` | Alto de la grilla | 30 |
| `mask` | Preset de máscara | (primer preset) |

**Qué se observa:**

- Formación de daemons (burbujas de activación estables)
- Movimiento direccional de los daemons
- Exclusión competitiva entre daemons cercanos
- Balance natural (~50% de neuronas activas)
- Convergencia ante manipulación externa

Las máscaras disponibles se cargan dinámicamente desde `core/masks.py`.
Ver [Modelo Neuronal](../core/README.md) para la nomenclatura.

---

## Experimentos planificados

| # | Nombre | Etapa | Descripción |
|---|--------|-------|-------------|
| 2 | **Dynamic SOM** | 2 | Mapa auto-organizativo con el modelo conexionista |
| 3 | **Motor y Nociceptor** | 3 | Modelos teóricos de salida motora e inhibición nociceptiva |
| 4 | **Tuning** | 4 | Optimización con algoritmos genéticos y scikit-learn |
| 5 | **Agentes Motores** | 5 | Agentes en mundo simulado (Aplysia, pez cebra) |

Ver [Hoja de Ruta](../../docs/STAGES.md) para el detalle de cada etapa.

---

## Cómo agregar un experimento

1. Crear un nuevo archivo en esta carpeta (ej. `dynamic_som.py`)
2. Heredar de `Experiment` (en `base.py`)
3. Implementar `setup()`, `step()`, `click()`, `reset()`
4. Registrar en `__init__.py`
5. El frontend lo descubre automáticamente vía la API

---

← Volver al [README](../../README.md)

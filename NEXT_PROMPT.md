# Tarea: Selector de conexionado + preview de máscara en el sidebar

## Contexto

En el Kohonen Lab, el selector de **conexionado** (`mask`) actualmente se muestra como una lista vertical de botones con nombre y descripción, dentro de un `div` con `maxHeight: 220px` y scroll. Hay 10 presets y la lista se está volviendo incómoda de usar.

Además, no hay forma de ver cómo es el patrón de conexión de una máscara antes de iniciar el experimento.

## Objetivo

Dos cambios relacionados, ambos puramente en el sidebar:

### Cambio 1: Convertir la lista de botones en un `<select>`

En `frontend/src/components/Sidebar.tsx`, reemplazar el bloque de botones de conexionado (el `div` con `maxHeight: 220px` que renderiza `selectedExp.masks.map(...)`) por un `<select>` estándar, con el mismo estilo que los otros selects del sidebar (`init_modes`, `balance_modes`).

El `<select>` debe:
- Mostrar `m.name` como texto de cada `<option>`, con `m.id` como `value`
- Cambiar `config.mask` al seleccionar
- Usar el mismo `inputStyle` que el resto de controles

Debajo del `<select>`, mostrar la descripción de la máscara seleccionada como texto pequeño (el campo `m.description` de `MaskPresetInfo`). Usar el mismo estilo de hint que ya tiene el control de balance (`fontSize: "0.65rem"`, `color: "#555"`).

### Cambio 2: Preview de conexionado con píxeles

Inmediatamente debajo del selector (y su descripción), agregar un mini canvas que visualice el patrón de conexión de la máscara seleccionada.

#### Cómo funciona la visualización

El backend ya genera un `weight_grid` cuando el inspector inspecciona una neurona. Se quiere algo análogo pero estático, calculado a partir de los offsets de la máscara, sin necesidad de un experimento activo.

**En el backend**, agregar un campo `preview_grid` a cada entrada de `get_mask_info()` en `backend/core/masks.py`. Es una grilla 2D pre-calculada de `float | None`:
- Tamaño fijo: 19×19 (para cubrir el mayor radio de offsets, que es 7 en `wide_hat` y `double_ring`)
- Centro en `(9, 9)` (índices, base 0)
- La celda central se marca con `999.0` (igual que el inspector marca la celda inspeccionada)
- Para cada offset `(dx, dy)` de cada dendrita:
  - `col = 9 + dx`, `row = 9 + dy`
  - Si cae dentro de la grilla: `grid[row][col] = peso_dendrita`
  - Si ya había un valor en esa celda, conservar el de mayor valor absoluto (evita sobreescribir exc con inh si se solapan)
- Celdas no tocadas: `None`

El tipo en la API ya existe: `weight_grid: (number | null)[][]` en `ConnectionsMessage`. Aquí es lo mismo pero como campo estático en la info de la máscara.

**En el frontend**, agregar a `MaskPresetInfo` en `frontend/src/types/index.ts`:
```ts
preview_grid: (number | null)[][];
```

Luego, en `Sidebar.tsx`, renderizar el preview. No importar `PixelCanvas` (tiene interactividad innecesaria). Crear un pequeño componente local `MaskPreview` que:
- Recibe `grid: (number | null)[][]`
- Usa un `<canvas>` con `useRef` + `useEffect`
- Tamaño visual: ancho = 100% del sidebar (220px aprox), altura proporcional (19×19 → cuadrado)
- Usa exactamente la misma función `weightToColor` que ya existe en `PixelCanvas.tsx` (extraerla o duplicarla en `Sidebar.tsx`)
- Fondo `#0a0a0a`, sin interactividad, sin borde interactivo
- Sin líneas de grilla (las celdas `null` simplemente quedan en `#111111`)

## Estructura de archivos a tocar

- `backend/core/masks.py` — agregar `preview_grid` a cada entrada de `MASK_PRESETS` y a `get_mask_info()`
- `frontend/src/types/index.ts` — agregar `preview_grid` a `MaskPresetInfo`
- `frontend/src/components/Sidebar.tsx` — reemplazar lista de botones por `<select>` + descripción + `MaskPreview` canvas

## Notas técnicas

- `get_mask_info()` ya filtra el campo `"mask"` al retornar. Agregar `preview_grid` como campo calculado al construir cada entrada del preset, o calcularlo lazy en `get_mask_info()`.
- Los offsets de algunas máscaras como `wide_hat` llegan hasta `r=7` (Chebyshev), por eso se necesita al menos radio 7 → grilla de 15×15 mínimo. Con 19×19 hay margen.
- `double_ring` tiene dos anillos con pesos distintos (`-1.0` y `-0.5`). Ambos deben aparecer en el preview con sus pesos correctos.
- El `MaskPreview` no necesita ser interactivo. No hay click, no hay drag, no hay tooltip.
- No tocar `PixelCanvas.tsx`. El `MaskPreview` es un componente local en `Sidebar.tsx`.
- No agregar tests para el preview (es visual). Sí asegurarse de que los tests existentes siguen pasando (el campo `preview_grid` nuevo no rompe nada porque `get_mask_info()` ya excluía el campo `mask`, y `preview_grid` se agrega igual).

## Verificación

1. `./venv/bin/pytest tests/ -v` — todos pasan
2. En el browser: Kohonen Lab → el selector de conexionado es un `<select>`, debajo aparece la descripción, debajo el mini canvas mostrando verde (exc) / rojo-morado (inh) con el centro en amarillo
3. Al cambiar de preset, el preview se actualiza instantáneamente (sin llamada al servidor)

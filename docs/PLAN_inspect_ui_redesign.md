# Inspect UI Redesign — Next Session Plan

## Estado actual (problemas)

1. **Overlay encima del canvas**: el `connectionMap` y el `weightOverlay` se renderizan sobre el grid real,
   haciendo ambos ilegibles al mismo tiempo (tissue verde + activaciones verdes = todo verde, sin diferencia).

2. **Info panel cortado en cajas pequeñas**: el overlay `inspectInfo` está posicionado `absolute top:8 right:8`
   dentro del `LayerBox`. Para una región output de 1×2, el box mide ~48×96px — el panel no cabe y se recorta.

3. **Bug: siempre hay que hacer dos clics**. Causa probable: `handleCellClick` captura `inspectMode` en un
   closure de `useCallback`. Si el usuario activa inspect mode y enseguida clica, el primer clic puede leer
   el valor stale de `inspectMode = false`. Fix: usar `useRef<boolean>` para `inspectMode` como fuente de
   verdad en el callback, manteniendo el state de React solo para re-render.

4. **Sin contexto visual**: no hay forma de ver qué región/neurona corresponde a qué overlay. Falta orientación.

---

## Diseño objetivo

### A. Paneles flotantes de pesos (en lugar de overlay sobre canvas)

Cuando se inspeciona la neurona (regionId, x, y):

- Aparece un nuevo **LayerBox flotante** con título `tissue · weights` encima/solapado con el box de tissue,
  mostrando el `connectionMap` como su propio canvas verde/purple independiente. El canvas real de tissue
  debajo sigue mostrando activaciones normales.
- Si hay `inputWeightGrid`, aparece otro **LayerBox flotante** sobre el box de input, con título `input · weights`.
- Posición inicial: superpuesto sobre el box correspondiente, desplazado unos píxeles arriba/derecha.
  Después el usuario puede arrastrarlo.
- Mismo sistema de `LayerBox` draggable/resizable ya existente — solo hay que añadir la clave en `boxes`.

```
boxes = {
  "tissue": { ... },           // siempre
  "input":  { ... },           // si existe
  "output": { ... },           // si existe
  "inspect:tissue": { ... },   // solo cuando inspeccionando — overlay weights del tissue
  "inspect:input":  { ... },   // solo cuando inspeccionando y hay inputWeightGrid
}
```

### B. Panel de info flotante con flecha

- Componente `InspectInfoPanel` — `position: absolute`, draggable (mismo patrón que LayerBox pero sin resize).
- Posición inicial: a la derecha del box de la región inspeccionada.
- Contenido: region id, coords, activation, tension, dendrites/synapses.
- **Flecha SVG** desde el panel hasta la celda inspeccionada en el canvas:
  - Capa SVG `position: absolute; inset: 0; pointer-events: none; zIndex: 25` sobre todo el scene.
  - Línea `<line>` o `<path>` calculada en tiempo real a partir de las posiciones de box + celda + panel.
  - La celda inspeccionada sigue teniendo el marker `999` en `PixelCanvas` (cuadrado amarillo).

### C. Toggle limpio (sin bug de doble clic)

```tsx
// En useExperiment.ts
const inspectModeRef = useRef(false);

const toggleInspectMode = useCallback(() => {
  const next = !inspectModeRef.current;
  inspectModeRef.current = next;
  setInspectMode(next);
  if (!next) {
    // limpiar todo
    send({ action: "uninspect" });
    ...
  }
}, [send]);

// En App.tsx
const handleCellClick = useCallback((x, y, regionId?) => {
  if (inspectModeRef.current) {   // <-- ref, no closure
    inspect(x, y, regionId);
  } else if (!regionId || regionId === tissueId) {
    applyBrush(x, y);
  }
}, [inspect, applyBrush, tissueId]);
```

Pero `inspectModeRef` vive en `useExperiment`, y `handleCellClick` en `App`. Dos opciones:
- Exponer el ref desde el hook: `inspectModeRef: React.RefObject<boolean>`
- O mover `handleCellClick` dentro del hook

Opción preferida: **exponer el ref** — mínimo cambio.

### D. Flujo de UX resultante

1. Usuario aprieta el botón de inspect en BrushPalette → modo ON.
2. Clica cualquier neurona en cualquier región → aparecen los paneles flotantes.
3. Puede arrastrar los paneles y seguir clicando otras neuronas (los paneles se actualizan en vivo).
4. Aprieta inspect de nuevo → modo OFF, todos los paneles `inspect:*` desaparecen del `boxes`.

---

## Archivos a modificar

| Archivo | Cambio |
|---|---|
| `useExperiment.ts` | Añadir `inspectModeRef`, exponer desde hook |
| `App.tsx` | Usar `inspectModeRef` en `handleCellClick` |
| `Scene.tsx` | Render condicional de `boxes["inspect:tissue"]` y `boxes["inspect:input"]`; SVG layer para flecha; `InspectInfoPanel` |
| `PixelCanvas.tsx` | Quitar soporte de `weightOverlay` y `connectionMap` del canvas directo — ahora son su propio panel |
| `LayerBox.tsx` | Sin cambios |
| nuevo: `InspectInfoPanel.tsx` | Panel draggable con info de neurona |

---

## Orden de implementación

1. Fix bug doble clic (ref) — 10 min, aislado, testeable inmediatamente
2. Crear `boxes["inspect:tissue"]` y `boxes["inspect:input"]` en Scene — los paneles aparecen posicionados
3. Quitar `weightOverlay`/`connectionMap` del canvas original
4. Crear `InspectInfoPanel` draggable
5. Añadir capa SVG con flecha

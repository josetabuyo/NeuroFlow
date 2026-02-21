# NeuroFlow

**Un modelo conexionista de la mente — más allá del lenguaje.**

Red de neuronas artificiales que busca el *daemon*: la unidad mínima de
procesamiento distribuido que emerge sin observador central, inspirada en la
neurociencia, los autómatas celulares y la filosofía de la consciencia.

---

## En una línea

> Tejido 2D de neuronas conectadas por sinapsis y dendritas donde daemons
> compiten, se estabilizan y auto-organizan — un camino conexionista hacia
> la emulación de la mente.

---

## ¿Qué es NeuroFlow?

NeuroFlow es un framework de autómatas neuronales conexionistas. Cada pixel de
la pantalla es una neurona. Las neuronas se conectan entre sí mediante dendritas
y sinapsis. El comportamiento emerge de reglas locales simples, sin controlador
central.

```
Sinapsis (peso ≥ 0)  →  Dendrita (peso ∈ [-1,1])  →  Neurona (activa/inactiva)
   reconoce patrón         AND fuzzy + inhibición         OR fuzzy competitivo
```

El proyecto no busca replicar lo que los LLMs ya resuelven (lenguaje), sino
atacar las áreas menos exploradas: **movimiento**, **percepción visual** y la
**profundidad del razonamiento** — lo que informalmente podríamos llamar
*intuición*.

---

## Documentación

La documentación está organizada en niveles de profundidad. Empezá por donde
te interese:

| Documento | Qué encontrás |
|-----------|---------------|
| **[Visión y Filosofía](docs/VISION.md)** | Qué es un daemon, por qué sin observador central, inspiraciones teóricas |
| **[Hoja de Ruta](docs/STAGES.md)** | Las 5 etapas del proyecto: Daemons → SOM → Motor/Nociceptor → Tuning → Agentes |
| **[Arquitectura Técnica](docs/ARCHITECTURE.md)** | Stack, diseño de clases, API, protocolo WebSocket, hosting |
| **[Referencias](docs/REFERENCES.md)** | Bibliografía completa con citas: Dennett, Hawkins, Kohonen, Kandel y más |
| **[Sobre el Autor](docs/AUTHOR.md)** | José Miguel Tabuyo — trayectoria, motivación y dedicatoria |

### Documentación cercana al código

| Documento | Qué encontrás |
|-----------|---------------|
| **[Modelo Neuronal](backend/core/README.md)** | Cómo funcionan Sinapsis, Dendrita, Neurona y Red en el código |
| **[Experimentos](backend/experiments/README.md)** | Qué hace cada experimento, cómo se configura, qué se observa |

---

## Stack

| Capa | Tecnología | Hosting |
|------|-----------|---------|
| Backend | Python 3.11+ / FastAPI / WebSocket / NumPy | Render.com (free) |
| Frontend | Vite / React 19 / TypeScript / HTML5 Canvas | Vercel (free) |

```
Frontend (React + Canvas)  ←WebSocket→  Backend (FastAPI)
         UI                                Red → Neurona → Dendrita → Sinapsis
```

---

## Fast Start

```bash
./start.sh
```

Levanta backend (`:8501`) y frontend (`:5173`) en paralelo.
Abrir **http://localhost:5173** en el navegador.

### Desde cero

```bash
# Backend (puerto 8501)
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8501

# Frontend (puerto 5173) — en otra terminal
cd frontend
npm install
npm run dev
```

### Tests

```bash
# Unitarios (backend)
cd backend && pytest -v

# E2E (frontend + backend, Playwright)
cd frontend
npx playwright install        # solo la primera vez
npm run test:e2e              # headless
npm run test:e2e:ui           # modo interactivo
```

---

## Origen

Este proyecto evoluciona de [RedJavaScript](https://github.com/), una
implementación 100% en navegador. NeuroFlow separa frontend (visualización)
de backend (cómputo), permitiendo escalar y desplegar como servicio web.

---

## Licencia

*Por definir.*

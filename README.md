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

## 🏗️ Estructura del Proyecto

```
CNA_Project/
├── README.md                    # Este archivo
├── PROYECTO_JUPYTER.md          # Documento completo con teoría y código
├── requirements.txt             # Dependencias pip
├── environment.yml              # Dependencias conda
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
│   └── robot/
│       ├── simple_robot.py      # Robot simulado
│       └── interface.py         # CNA-Robot bridge
├── experiments/
│   ├── aplysia_reflex.py        # Reflejo condicionado simple
│   ├── zebrafish_navigation.py  # Navegación espacial
│   └── rat_memory_replay.py     # Memory replay
└── tests/
    ├── test_neuron.py
    ├── test_automaton.py
    └── test_learning.py
```

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

## 📖 Documentación Completa

Para entender la filosofía y fundamentos científicos completos, lee:

👉 **[PROYECTO_JUPYTER.md](PROYECTO_JUPYTER.md)** (Documento principal con toda la teoría, código y experimentos)

---

## 📧 Contacto

José Tabuyo - [@tutwitter](https://twitter.com/tutwitter)

Proyecto Link: [https://github.com/JoseTabuyo/RedJavaScript](https://github.com/JoseTabuyo/RedJavaScript)

---

**⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!**

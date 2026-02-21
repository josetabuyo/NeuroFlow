# Referencias

Bibliografía organizada por área temática. Cada entrada incluye la
relevancia para NeuroFlow.

---

## Filosofía de la mente y consciencia

### Dennett, D. C. (1991). *Consciousness Explained*. Little, Brown and Company.

La obra central que inspira la arquitectura de NeuroFlow. Dennett
demuestra que no existe un "teatro cartesiano" — un observador central
donde se proyecta la consciencia — y propone en su lugar un modelo de
**Múltiples Borradores** (*Multiple Drafts*) donde procesos paralelos
compiten para emerger. Adopta el modelo Pandemonium de Selfridge e
introduce la **máquina Joyciana**: una entidad virtual que emerge de la
competencia distribuida de daemons.

> *"There is no single, definitive 'stream of consciousness,' because
> there is no central Headquarters, no Cartesian Theater where 'it all
> comes together'."*

### Selfridge, O. G. (1959). "Pandemonium: A Paradigm for Learning." *Symposium on the Mechanization of Thought Processes*.

El modelo original de daemons: procesos semi-independientes que operan
en paralelo y compiten entre sí. Fue uno de los primeros programas
conexionistas, con modelos evolutivos donde las fuerzas de conexión
evolucionan con el tiempo. NeuroFlow implementa literalmente esta
dinámica en su tejido neuronal.

### Hofstadter, D. R. & Dennett, D. C. (Eds.). (1981). *The Mind's I: Fantasies and Reflections on Self and Soul*. Basic Books. ISBN 978-0-465-03091-0.

Antología que reúne ensayos de Borges, Dawkins, Searle y otros sobre
inteligencia artificial, consciencia, la naturaleza del yo y el alma.
Fue el libro que introdujo al autor de NeuroFlow en la filosofía de la
mente y lo conectó con el trabajo de Dawkins y Dennett.

---

## Biología evolutiva y memética

### Dawkins, R. (1976). *The Selfish Gene*. Oxford University Press. ISBN 0-19-929114-4.

Introduce el concepto de **memes** — unidades de información cultural
que se replican y evolucionan por selección natural, análogas a los
genes. El capítulo 11, "Memes: the new replicators", es fundacional
para la memética. La visión de la realidad como algo evolutivo,
iterativo y distribuido informa profundamente la filosofía de NeuroFlow.

---

## Neurociencia computacional

### Hawkins, J. & Blakeslee, S. (2004). *On Intelligence*. Times Books. ISBN 978-0-8050-7456-7.

Propone que el neocórtex es una **máquina de predicción**: aprende un
modelo jerárquico del mundo y predice entradas futuras. Fundó
**Numenta** para desarrollar HTM (Hierarchical Temporal Memory), un
enfoque biológicamente informado basado en la fisiología de neuronas
piramidales del neocórtex. NeuroFlow se inspira fuertemente en la idea
de predicción jerárquica y en la organización columnar del neocórtex.

### Kandel, E. R. (2001). "The Molecular Biology of Memory Storage: A Dialogue Between Genes and Synapses." *Bioscience Reports*, 21(5), 507-522. Nobel Lecture.

Los estudios de Kandel sobre ***Aplysia californica*** descubrieron los
mecanismos sinápticos del aprendizaje y la memoria: la memoria a largo
plazo involucra remodelación sináptica y crecimiento de nuevas sinapsis.
*Aplysia* es modelo objetivo para la Etapa 5 (Agentes Motores) por su
sistema nervioso simple (~20.000 neuronas) que permite atribuir cambios
conductuales a neuronas y sinapsis específicas.

### Kandel, E. R., Schwartz, J. H. & Jessell, T. M. (2000). *Principles of Neural Science* (4th ed.). McGraw-Hill.

Referencia comprehensiva sobre neurociencia, incluyendo nociceptores,
circuitos motores y organización cortical.

---

## Modelos de auto-organización

### Kohonen, T. (1990). "The Self-Organizing Map." *Proceedings of the IEEE*, 78(9), 1464-1480.

El **mapa auto-organizativo** (SOM) es una red neuronal donde las
células se sintonizan a patrones de entrada mediante aprendizaje no
supervisado, creando representaciones espacialmente organizadas análogas
a los mapas topográficos del cerebro. Usa interacciones laterales
competitivas con patrón de **sombrero mexicano** (excitación local,
inhibición lateral). NeuroFlow implementa este principio con sus
máscaras Daemon.

### Hubel, D. H. & Wiesel, T. N. (1962). "Receptive Fields, Binocular Interaction and Functional Architecture in the Cat's Visual Cortex." *Journal of Physiology*, 160(1), 106-154.

Descubrieron la organización de campos receptivos en el córtex visual
del gato — la base biológica del sombrero mexicano y de la inhibición
lateral. Premio Nobel de Fisiología/Medicina (1981). Su observación de
la naturaleza es el fundamento biológico de los conectados Daemon en
NeuroFlow.

---

## Autómatas celulares y computación

### Wolfram, S. (2002). *A New Kind of Science*. Wolfram Media.

Estudio sistemático de autómatas celulares elementales. Demuestra que
reglas extremadamente simples (como la Rule 110) pueden realizar
computación universal. NeuroFlow usa autómatas de Wolfram como banco de
pruebas: si el modelo conexionista puede reproducir las 256 reglas
elementales, valida la expresividad del sistema Sinapsis→Dendrita→Neurona.

---

## Circuitos motores y locomoción

### Warp, E. et al. (2012). "Emergence of patterned activity in the developing zebrafish spinal cord." *Current Biology*, 22(2), 93-102.

### Svara, F. et al. (2023). "Molecular blueprints for spinal circuit modules controlling locomotor speed in zebrafish." *Nature Neuroscience*.

El **pez cebra** (*Danio rerio*) es modelo para la Etapa 5 por su
sistema nervioso transparente y circuitos motores espinales bien
caracterizados. Los modelos computacionales de circuitos locomotores
en pez cebra muestran cómo la diversidad molecular se traduce en
módulos funcionales que controlan velocidad y tipo de movimiento.

---

## Reconocimiento de patrones y clasificación

### Fukunaga, K. (1990). *Introduction to Statistical Pattern Recognition* (2nd ed.). Academic Press.

### Duda, R. O., Hart, P. E. & Stork, D. G. (2001). *Pattern Classification* (2nd ed.). Wiley. ISBN 0-471-05669-3.

Textos fundamentales sobre clasificación en hiperespacios. Describen la
progresión de hipótesis de clasificación: esferas → elipses → elipsoides
→ elipsoides correlacionados → formas complejas. Este marco teórico
informó el trabajo del autor en el GIAR (UTN) sobre reconocimiento de
texto manuscrito y descriptores de imagen (momentos de Hu).

*Nota: El autor realizó este estudio de forma independiente y descubrió
posteriormente que la literatura existente confirmaba sus hallazgos.*

---

## Dolor y nociceptores

### Melzack, R. & Wall, P. D. (1965). "Pain Mechanisms: A New Theory." *Science*, 150(3699), 971-979.

La **teoría de compuerta del dolor** (gate control theory) propone que
la percepción del dolor no es un canal directo sino un sistema donde
señales excitatorias (fibras C, nociceptivas) e inhibitorias (fibras
Aβ, mecánicas) compiten en el asta dorsal de la médula espinal. Este
modelo de competencia entre excitación e inhibición es directamente
análogo a la dinámica de daemons en NeuroFlow.

---

## Redes convolucionales y visualización

### Mordvintsev, A. et al. (2015). "Inceptionism: Going Deeper into Neural Networks." *Google Research Blog*.

**Deep Dream** demuestra que una red convolucional entrenada para
clasificar imágenes contiene suficiente información para *generarlas*.
Las capas extraen características progresivamente más complejas — desde
bordes hasta objetos completos. La capacidad de organizar imágenes por
similitud es análoga a lo que hace un SOM, conectando la visión
artificial moderna con el enfoque conexionista de NeuroFlow.

---

## Divulgación

### DotCSV (Carlos Santana Vega). Canal de YouTube.

Canal de divulgación de inteligencia artificial en español (~1M
suscriptores). Sus explicaciones sobre redes convolucionales y Deep
Dream iluminaron la conexión entre la organización por similitud en
redes profundas y los mapas auto-organizativos.

URL: [youtube.com/@DotCSV](https://www.youtube.com/@DotCSV)

---

← Volver al [README](../README.md)

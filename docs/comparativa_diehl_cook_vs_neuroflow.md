# Comparativa: Diehl & Cook (2015) vs NeuroFlow

Paper: *"Unsupervised learning of digit recognition using spike-timing-dependent plasticity"*
Frontiers in Computational Neuroscience, Vol. 9, 2015 | DOI: 10.3389/fncom.2015.00099
Autores: Peter U. Diehl, Matthew Cook — ETH Zurich / University of Zurich
1,300 citas | 112k views

---

## 1. Resumen del paper de Diehl & Cook

Red neuronal spiking (SNN) de dos capas que aprende a reconocer digitos MNIST de forma **no supervisada** usando STDP (spike-timing-dependent plasticity). Alcanza 95% de accuracy con 6400 neuronas excitatorias. No usa backpropagation, ni senales de ensenanza, ni preprocesamiento de las imagenes.

Mecanismos clave:
- Neuronas LIF (leaky integrate-and-fire) con sinapsis basadas en conductancia
- STDP con ventana temporal exponencial
- Inhibicion lateral via neuronas inhibitorias dedicadas
- Umbral adaptativo (homeostasis/plasticidad intrinseca)

---

## 2. Tabla comparativa lado a lado

| Dimension | Diehl & Cook (2015) | NeuroFlow |
|---|---|---|
| **Modelo neuronal** | Leaky Integrate-and-Fire (LIF). Potencial de membrana continuo, ecuaciones diferenciales. | Activacion binaria (0/1). Umbral sobre tension calculada por logica fuzzy. |
| **Sinapsis** | Conductance-based. El efecto de un spike depende del voltaje actual de membrana. | Similitud fuzzy: `1 - \|peso - entrada\|`. Resultado continuo [0,1]. |
| **Representacion de valores** | Continuo: voltaje de membrana, frecuencias de disparo, conductancias. | Binario: neuronas ON/OFF. Pesos continuos [0,1]. |
| **Topologia de la red** | **Sin topologia espacial**. Las neuronas excitatorias no tienen posicion ni vecindad. Cada una es independiente. | **Grid 2D con topologia local**. Las neuronas tienen coordenadas (x,y) y conectividad local definida por mascaras (daemons). |
| **Arquitectura** | 2 capas feedforward: input (784) → excitatory (N) ← inhibitory (N). | Tejido 2D (tissue) + region de entrada. Cada neurona del tejido recibe input Y tiene conexiones locales. |
| **Inhibicion lateral** | **Global**: cada neurona inhibitoria se conecta a TODAS las excitatorias (menos su fuente). Winner-take-all implicito. | **Local**: anillo inhibitorio alrededor de cada neurona (sombrero mexicano). Multiples dendritas sectoriales. |
| **Aprendizaje** | STDP: cambio de peso basado en la diferencia temporal entre spikes pre y post-sinapticos. 4 variantes probadas. | Hebbian modulado por tension: `dW = lr * tension * (source - weight)`. Todas las neuronas aprenden simultaneamente. |
| **Homeostasis** | Umbral adaptativo θ: se incrementa con cada spike, decae exponencialmente. Impide que una neurona domine. | Spike Frequency Adaptation: `max_active_steps` + periodo refractario. Fuerza apagado despues de N pasos activos. |
| **Input encoding** | Poisson spike trains. Frecuencia proporcional a intensidad del pixel. 350ms por estimulo. | Proyeccion directa de pixeles binarios. Multiples frames por caracter con ruido. |
| **Tarea** | Clasificacion MNIST (10 clases de digitos). | Auto-organizacion ante stream visual (caracteres ASCII con ruido). |
| **Supervision** | No supervisado para entrenamiento. Las etiquetas solo se usan post-entrenamiento para asignar clases a neuronas. | Completamente no supervisado. No hay asignacion de clases. |
| **Escala** | 100 a 6400 neuronas excitatorias. | Grid tipico de 50x50 = 2500 neuronas de tejido + region de entrada. |

---

## 3. Similitudes fundamentales

### 3.1 Aprendizaje no supervisado con competencia local

Ambos sistemas aprenden sin etiquetas. La competencia emerge de mecanismos inhibitorios que fuerzan a las neuronas a especializarse en diferentes patrones.

- **Diehl & Cook**: la inhibicion global (all-to-all) crea un winner-take-all estricto. Cuando una neurona dispara, suprime a todas las demas.
- **NeuroFlow**: la inhibicion local (sombrero mexicano) crea competencia entre neuronas cercanas. Los "daemons" emergen como unidades funcionales.

### 3.2 Mecanismo anti-dominancia

Ambos necesitan impedir que una neurona (o grupo) domine toda la red:

- **Diehl & Cook**: umbral adaptativo θ. Cuanto mas dispara una neurona, mas alto se pone su umbral, dando oportunidad a otras.
- **NeuroFlow**: `max_active_steps` + periodo refractario. Si una neurona esta activa demasiados pasos, se fuerza su apagado.

Biologicamente, ambos modelan **Spike Frequency Adaptation (SFA)**, pero de formas diferentes:
- θ adaptativo = modelo de corrientes de potasio lentas (gradual)
- max_active_steps = modelo de burst limitado + periodo refractario absoluto (discreto)

### 3.3 Pesos sinapticos como campos receptivos

En ambos sistemas, los pesos sinapticos de entrada convergen hacia prototipos de los patrones presentados:

- **Diehl & Cook**: los pesos de 784 sinapsis (una por pixel) se pueden reorganizar en matrices 28x28 que visualmente muestran digitos aprendidos (ver Figure 2A del paper).
- **NeuroFlow**: los pesos de las sinapsis de entrada (input_weight_grid) deberian converger hacia las formas de los caracteres proyectados.

### 3.4 Ruido como regularizador

- **Diehl & Cook**: los Poisson spike trains introducen variabilidad en cada presentacion del mismo digito.
- **NeuroFlow**: white noise y shift noise se aplican a los frames de entrada.

---

## 4. Diferencias criticas

### 4.1 Topologia: la diferencia mas profunda

Esta es la divergencia fundamental. Diehl & Cook no tienen topologia espacial: sus neuronas excitatorias son un conjunto no ordenado. No hay concepto de "neurona vecina" en la capa excitadora.

NeuroFlow es **inherentemente topologico**. Cada neurona existe en un grid 2D con coordenadas, y su conectividad local (daemon) define su vecindad funcional. Esto tiene consecuencias enormes:

1. **Diehl & Cook no es un SOM**: aunque usan competencia, no hay preservacion de topologia. Dos neuronas que aprenden digitos similares pueden estar en cualquier posicion.
2. **NeuroFlow es un SOM genuino**: la inhibicion local + excitacion local fuerza que neuronas vecinas respondan a patrones similares. La topologia del input deberia mapearse a la topologia del tejido.

Esto hace que NeuroFlow sea conceptualmente mas cercano a Kohonen (1982) que a Diehl & Cook, pero con mecanismos biologicos en vez de algoritmos abstractos.

### 4.2 Inhibicion: global vs local

| Aspecto | Diehl & Cook (global) | NeuroFlow (local) |
|---|---|---|
| Alcance | Cada neurona inhibe a TODAS las demas | Cada neurona inhibe solo dentro de un radio |
| Selectividad | Winner-take-all estricto: solo 1 neurona gana | Multiples "ganadores" (daemons) pueden coexistir |
| Escalabilidad | O(N^2) conexiones inhibitorias | O(N * k) donde k = vecinos en el anillo |
| Emergencia | No hay patrones espaciales emergentes | Daemons: clusters auto-organizados de actividad |

La inhibicion global de Diehl & Cook es elegante pero biologicamente cuestionable a gran escala: requiere que una interneurona envie axones a miles de neuronas distantes. La inhibicion local de NeuroFlow es mas plausible anatomicamente.

### 4.3 Modelo neuronal: LIF vs binario-fuzzy

Diehl & Cook usan un modelo LIF estandar con ecuaciones diferenciales:
```
τ dV/dt = (E_rest - V) + g_e(E_exc - V) + g_i(E_inh - V)
```

NeuroFlow usa un modelo radicalmente diferente:
```
similitud = 1 - |peso - entrada|
promedio_dendrita = mean(similitudes)
contribucion = promedio_dendrita * peso_dendrita
tension = fuzzy_OR(contribuciones)  // max(pos) + min(neg)
activacion = tension > umbral
```

Diferencias clave:
- **LIF es temporal**: la membrana integra spikes a lo largo del tiempo. Un spike aislado puede no ser suficiente, se necesita acumulacion.
- **NeuroFlow es instantaneo**: la activacion se calcula en un solo paso a partir del estado actual. No hay integracion temporal dentro de un step.
- **LIF usa diferencia de potencial**: las corrientes dependen del voltaje actual (conductance-based).
- **NeuroFlow usa similitud**: la sinapsis mide cuanto se parece el peso al input, no suma corrientes.

### 4.4 Aprendizaje: STDP vs Hebbian por tension

**STDP (Diehl & Cook):** el cambio de peso depende de la **temporalidad relativa** de los spikes:
- Pre antes de post → potenciacion (Long-Term Potentiation)
- Post antes de pre → depresion (Long-Term Depression)
- La magnitud decae exponencialmente con el intervalo temporal

**Hebbian por tension (NeuroFlow):** el cambio depende del **estado actual** de la neurona:
```
dW = lr * tension * (source_value - weight)
```
- Tension positiva: el peso se acerca al valor de la fuente
- Tension negativa: el peso se aleja del valor de la fuente
- No hay dependencia temporal entre eventos

Implicacion: STDP es un mecanismo de **correlacion temporal** (aprender causalidad). El Hebbian de NeuroFlow es un mecanismo de **correlacion espacial** (aprender coactivacion).

### 4.5 Estructura dendritica

**Diehl & Cook**: no modelan dendritas. Cada sinapsis contribuye directamente a la conductancia de la neurona. Es un modelo de un solo compartimento.

**NeuroFlow**: modelo multi-dendritico. Cada neurona tiene multiples dendritas, cada una con su peso y conjunto de sinapsis. Las dendritas operan como sub-unidades computacionales:
- Una dendrita excitatoria agrupa las sinapsis de vecinos cercanos
- Multiples dendritas inhibitorias (sectoriales) agregan las sinapsis del anillo lejano
- Una dendrita de entrada recibe las sinapsis del input

Esto es biologicamente mas sofisticado: las dendritas reales son subunidades de procesamiento con no-linearidades locales (Poirazi & Mel, 2001).

---

## 5. Lo que NeuroFlow tiene y Diehl & Cook no

1. **Topologia espacial explicita**: las neuronas estan en un grid y la conectividad depende de la distancia. Esto permite auto-organizacion topografica real.

2. **Daemons como unidades funcionales emergentes**: no existen en Diehl & Cook. Los clusters de actividad con centro excitatorio + corona inhibitoria son un fenomeno emergente unico de NeuroFlow.

3. **Dendritas como subunidades computacionales**: la separacion en dendritas excitatorias, inhibitorias y de entrada permite computacion dendritica que LIF plano no puede hacer.

4. **Mascaras de conectividad configurables**: la biblioteca de mascaras (Mexican hat, gradual, ring, sparse) permite explorar sistematicamente como la geometria de conexion afecta la dinamica. Diehl & Cook tienen una sola arquitectura fija.

5. **Visualizacion en tiempo real**: NeuroFlow permite observar la dinamica de activacion, tension y pesos evolucionando en vivo. Diehl & Cook solo muestran resultados post-entrenamiento.

6. **Input temporal con gaps**: la alternancia caracter-gap con supresion sacadica del aprendizaje no tiene equivalente en Diehl & Cook, que presentan un estimulo a la vez sin transiciones.

---

## 6. Lo que Diehl & Cook tienen y NeuroFlow aun no

1. **STDP con ventana temporal**: el aprendizaje basado en temporalidad relativa de spikes es mas biologicamente plausible y captura causalidad. El Hebbian de NeuroFlow no tiene este aspecto temporal.

2. **Winner-take-all robusto**: la inhibicion global + umbral adaptativo garantiza que cada neurona se especialice en un patron diferente. NeuroFlow aun lucha con el "techo plano" donde muchas neuronas aprenden lo mismo.

3. **Evaluacion cuantitativa**: 95% accuracy en MNIST con benchmark estandar. NeuroFlow no tiene una metrica equivalente para medir calidad de la auto-organizacion.

4. **Modelo de membrana con integracion temporal**: el LIF integra spikes a lo largo del tiempo, lo que permite robustez al ruido. El modelo instantaneo de NeuroFlow es mas sensible a fluctuaciones frame-a-frame.

5. **Escalabilidad probada**: testaron 100 → 6400 neuronas con mejora monotona. NeuroFlow no ha explorado sistematicamente el efecto de escala.

6. **Sinapsis basadas en conductancia**: el efecto de un spike depende del voltaje actual, creando no-linearidades realistas. Las sinapsis fuzzy de NeuroFlow son lineales en su operacion.

---

## 7. Oportunidades de cruce

### 7.1 Incorporar umbral adaptativo de Diehl & Cook en NeuroFlow

El mecanismo de θ adaptativo podria complementar el `max_active_steps`:
```
umbral_efectivo = umbral_base + theta
theta += delta_theta (cada vez que dispara)
theta *= decay (cada step)
```

Esto daria una degradacion gradual en vez del corte abrupto del periodo refractario. Ambos mecanismos podrian coexistir: θ para adaptacion suave, `max_active_steps` como safety net.

### 7.2 Agregar temporalidad al aprendizaje de NeuroFlow

Una version simplificada de STDP para el modelo binario:
- Mantener un "trace" por neurona que decae cada step
- El trace sube cuando la neurona se activa
- El aprendizaje usa el trace pre-sinaptico: neuronas que fueron activas recientemente contribuyen mas

Esto capturaria parte de la causalidad temporal sin necesitar un modelo LIF completo.

### 7.3 Winner-take-all local

Adaptar la idea de Diehl & Cook pero hacerla local: dentro de cada daemon, solo la neurona con mayor tension aprende de la dendrita de entrada. Esto resolveria el problema del "techo plano" sin perder la topologia.

### 7.4 Metrica de evaluacion para NeuroFlow

Definir una metrica de "calidad de mapa" inspirada en SOMs clasicos:
- **Quantization error**: diferencia media entre input y el campo receptivo de la BMU.
- **Topographic error**: proporcion de veces que la BMU y la segunda BMU no son vecinas en el grid.

---

## 8. Contexto historico: donde encaja cada trabajo

```
Kohonen (1982)          Diehl & Cook (2015)         NeuroFlow (2025-2026)
SOM clasico             SNN + STDP                  Daemons + Fuzzy + Grid
│                       │                           │
├─ Algoritmo abstracto  ├─ Biologico pero sin       ├─ Topologico con
│  con BMU + vecindad   │  topologia espacial       │  mecanismos emergentes
│                       │                           │
├─ No spiking           ├─ LIF spiking              ├─ Binario con dendritas
│                       │                           │
├─ Deterministico       ├─ Estocastico (Poisson)    ├─ Ruido configurable
│                       │                           │
└─ Feedforward puro     └─ Inhibicion global        └─ Inhibicion local
                           (sin recurrencia              (sombrero mexicano
                            excitatoria)                  + recurrencia local)
```

NeuroFlow ocupa un nicho unico: toma la topologia de Kohonen, la biologia de las SNNs, y agrega estructura dendritica y dinamicas emergentes (daemons) que ninguno de los otros tiene. El reto principal es hacer que el aprendizaje converja tan robustamente como en Diehl & Cook.

---

## 9. Recomendacion: que leer del paper con detenimiento

1. **Section 2.4 (Homoeostasis)**: El mecanismo de umbral adaptativo θ. Es la pieza que mas directamente podria mejorar NeuroFlow.

2. **Figure 2A**: Los campos receptivos aprendidos. Comparar visualmente con tus `input_weight_grid` para ver si NeuroFlow converge a prototipos similares.

3. **Figure 2B**: Performance vs numero de neuronas. Sirve como benchmark: si NeuroFlow no mejora con mas neuronas, algo en la arquitectura lo impide.

4. **Section 4.2**: La discusion sobre como la competencia + homeostasis + STDP se complementan. Los tres ingredientes son necesarios — ninguno funciona solo.

5. **Table 1**: Comparativa con otros SNN. Da contexto de donde esta el estado del arte.

---

*Documento generado para analisis interno de NeuroFlow.*
*Paper original: https://doi.org/10.3389/fncom.2015.00099*

# Analisis: Problemas de Aprendizaje en el Dynamic SOM

## 1. Diagnostico del problema central

Los Daemons se vuelven estaticos y los pesos sinapticos de entrada permanecen ruidosos porque el sistema tiene **varios problemas compuestos** que impiden la diferenciacion. Los analizo en orden de impacto.

---

## 2. Problema critico: El aprendizaje NO se apaga durante el gap

En `step()`, el flujo es:

```
generate_and_project() -> procesar() -> learn() -> avanzar frame
```

El `learn()` se ejecuta en **cada step**, incluyendo durante la fase de gap entre caracteres. Esto es devastador:

- **Con `inter_char_noise=True` (default):** durante el gap se proyecta ruido blanco puro. Las neuronas aprenden hacia patrones aleatorios, deshaciendo parcialmente lo aprendido del caracter anterior.
- **Con `inter_char_noise=False`:** se proyecta imagen negra (todos ceros). Las neuronas con tension positiva mueven sus pesos de entrada hacia 0.0, que tampoco es el patron deseado.

En ambos casos, el gap **corrompe los pesos** en vez de ser un momento de descanso.

### Analogia biologica: Supresion sacadica

Tu intuicion sobre los movimientos sacadicos es correcta y esta respaldada por la neurociencia. Durante las sacadas oculares:

1. La corteza visual V1 recibe senales **extrarretinianas** que suprimen la sensibilidad visual (Nature Communications, 2015).
2. La supresion comienza ~75ms antes del inicio de la sacada.
3. El mecanismo es **divisivo** (multiplicativo, gain control), no simplemente aditivo.
4. Esto evita que el sistema procese y "aprenda" de las imagenes borrosas del movimiento.

**Traduccion al codigo:** durante el gap, el aprendizaje deberia desactivarse completamente. El procesamiento (procesar()) puede seguir activo para que los daemons se reconfiguren, pero learn() no deberia ejecutarse.

### Solucion inmediata

```python
def step(self):
    self._generate_and_project()
    self.brain_tensor.procesar()
    
    # NO aprender durante el gap (supresion sacadica)
    if self.learning_enabled and not self._in_gap and self.brain_tensor is not None:
        self.brain_tensor.learn(lr=self.learning_rate)
```

---

## 3. Problema del techo plano: Aprendizaje global sin winner-take-all

En un SOM clasico de Kohonen, el aprendizaje solo afecta a la **neurona ganadora** (BMU) y sus vecinas dentro de un radio de vecindad. En tu implementacion:

```python
def learn(self, lr):
    source_vals = self.valores[self.indices_fuente]
    tension = self.tensiones[:NR].unsqueeze(1)
    delta = lr * tension * (source_vals - self.pesos_sinapsis)
```

La tension modula el aprendizaje para TODAS las neuronas simultaneamente. Si el paisaje de tension es plano (muchas neuronas con tension similar), todas aprenden lo mismo. Esto causa exactamente lo que describes: "siempre se pisan los patrones de A y B en todas las neuronas".

### Por que el paisaje es plano

Los pesos sinapticos de entrada se inicializan con `random.uniform(0.2, 1.0)`. Con la funcion sinaptica `1 - |peso - entrada|`:

- Si el peso es ~0.6 (media del rango) y la entrada es 0 o 1, la similitud es ~0.4 en ambos casos.
- Todas las neuronas ven similitudes parecidas al principio.
- El promedio de estas similitudes por dendrita de entrada es similar para todos.
- La tension de cada neurona depende mas de los daemons locales que de la entrada.

Resultado: el input no logra crear gradientes suficientes para diferenciar neuronas.

### Opciones para resolver

**Opcion A:** Implementar un mecanismo de winner-take-all explicito donde solo las top-K neuronas con mayor similitud al input aprendan (o las neuronas dentro del daemon ganador).

**Opcion B:** Aumentar el peso de la dendrita de entrada (`input_dendrite_weight`) para que domine sobre los daemons en la determinacion de la tension, al menos inicialmente.

**Opcion C:** Usar inicializacion no uniforme de los pesos de entrada, de modo que diferentes regiones del tejido ya tengan "preferencias" iniciales distintas.

---

## 4. Hipotesis del techo plano del Daemon

Tu observacion es precisa: si la campana del sombrero mexicano tiene un techo plano, no hay una unica "punta" ganadora. Esto ocurre porque:

1. La dendrita excitatoria (Moore r=3, 48 vecinos) agrupa muchas sinapsis. El promedio de 48 valores binarios tiene varianza baja, lo que aplana el gradiente.
2. Con pesos aleatorios y entradas binarias, `1 - |w - input|` promediado sobre 48 sinapsis converge al mismo valor para todos los centros.
3. El gap silencioso entre la zona excitatoria y la inhibitoria (r=4 en `deamon_3_en_50`) puede no ser suficiente para crear selectividad espacial en presencia de la entrada.

### Relevancia de la literatura

El paper "Self-organizing maps: stationary states, metastability and convergence rate" (Springer, 1993) documenta exactamente este fenomeno: **estados metaestables** donde el mapa queda atrapado. La solucion propuesta es usar funciones de vecindad convexas sobre un rango amplio para evitar estos estados.

---

## 5. Tu hipotesis: Neuronas con duracion maxima de activacion

### Base biologica: Spike Frequency Adaptation (SFA)

Tu idea tiene un fundamento biologico solido y bien documentado:

1. **Spike Frequency Adaptation** (Scholarpedia): Las neuronas corticales **no pueden disparar indefinidamente**. Cuando reciben estimulacion sostenida, su frecuencia de disparo decrece progresivamente. Los mecanismos son:
   - Corrientes de potasio activadas por calcio (I_AHP): cada spike acumula calcio intracelular, que activa canales de potasio hiperpolarizantes.
   - Corrientes de potasio lentas (I_M, I_Ks): se activan con despolarizacion sostenida.
   
2. **Duracion limitada de bursts** (CUNY, 2018): Las neuronas piramidales de capas II/III pueden generar bursts de alta frecuencia (>300 Hz) pero **no pueden sostenerlos**. La duracion esta limitada por la interaccion entre after-hyperpolarizations (AHPs) y after-depolarizations (ADPs).

3. **Fatiga neural** (Nature, 2023): La estimulacion repetida causa disrupciones funcionales locales. Las areas que responden mas intensamente son las mas afectadas.

4. **Habituacion multi-escala** (Frontiers, 2022): La plasticidad cortical durante la habituacion ocurre en escalas de **segundos, minutos y dias**. Los mecanismos de corto plazo (los relevantes para tu caso) involucran neuronas inhibitorias de parvalbumina.

### Tu propuesta es correcta: las neuronas NO quedan activas continuamente

En la biologia:
- Una neurona cortical tipica puede sostener actividad por decenas a cientos de milisegundos antes de que la adaptacion reduzca significativamente su tasa de disparo.
- En tu modelo discreto, esto se traduce a: **una neurona activa solo puede permanecer activa durante N pasos consecutivos** antes de forzar una desactivacion.

### Que resolveria esto en tu modelo

1. **Rompe la estaticidad de los daemons**: Si una neurona solo puede estar activa 5 pasos, el daemon se ve forzado a "rotar" o "migrar". Esto crea diversidad en los estados que explora.
2. **Habilita la competencia**: Cuando las neuronas del centro del daemon se apagan por fatiga, las neuronas vecinas (que estaban inhibidas) tienen oportunidad de activarse, creando un movimiento natural.
3. **Interaccion con dendritas lejanas**: Las dendritas de entrada (lejanas, del input) podrían sesgar hacia donde migra el daemon cuando sus neuronas centrales se fatigan. Esto es exactamente el mecanismo que necesitas para que el input guie la organizacion.

### Implementacion propuesta

Un nuevo parametro `max_active_steps` (default: 5). Agregar un contador por neurona que track cuantos pasos consecutivos ha estado activa. Cuando alcanza el maximo, forzar desactivacion por 1 paso (periodo refractario).

```python
# En BrainTensor.__init__:
self.active_counts = torch.zeros(self.N, device=device)
self.max_active_steps = max_active_steps  # default 5
self.refractory = torch.zeros(self.N, dtype=torch.bool, device=device)

# En procesar(), despues de calcular nuevos_valores:
# Incrementar contador para neuronas activas
self.active_counts = torch.where(
    nuevos_valores > 0.5,
    self.active_counts + 1,
    torch.zeros_like(self.active_counts)
)

# Forzar desactivacion si exceden max_active_steps
fatigued = self.active_counts >= self.max_active_steps
nuevos_valores = torch.where(fatigued, torch.zeros_like(nuevos_valores), nuevos_valores)
self.active_counts = torch.where(fatigued, torch.zeros_like(self.active_counts), self.active_counts)
```

---

## 6. Relacion dendritas lejanas vs cercanas

Tu observacion sobre convulsiones cuando el peso de dendrita lejana supera al de la cercana es consistente con la teoria de estabilidad de redes neuronales. El punto clave:

- **Dendritas cercanas (daemons):** crean la estructura espacial (sombrero mexicano). Funcionan bien porque la regla de conectividad es fija y geometrica.
- **Dendritas lejanas (input):** traen informacion externa que deberia modular, no dominar, la dinamica local.

Si `input_dendrite_weight > deamon_exc_weight`, la entrada domina sobre la estructura local y el tejido se convierte en un "espejo" del input en vez de un mapa auto-organizado.

### Pre-entrenamiento de sinapsis cercanas

Tu idea de inicializar las sinapsis cercanas de forma no ruidosa tiene sentido. Las sinapsis del daemon no necesitan ser aleatorias porque su funcion es geometrica (detectar actividad local). Podrias:

1. Inicializarlas todas a 0.5 (neutral) en vez de random.
2. Correr N pasos sin input (solo daemons) para que se estabilicen.
3. Luego conectar el input y empezar el entrenamiento.

Sin embargo, hay que tener cuidado: si las sinapsis cercanas ya son perfectas, la regla de aprendizaje hebiana no las modificara (porque `source - weight` sera pequeno cuando hay match). El riesgo es que el daemon quede rigido y no pueda ser guiado por el input.

---

## 7. Investigadores trabajando en temas similares

### Spiking SOMs con periodos refractarios
- **Diehl & Cook (2015)**: "Unsupervised learning of digit recognition using spike-timing-dependent plasticity" (Frontiers in Computational Neuroscience). Implementaron SOMs con neuronas spiking que incluyen periodos refractarios y STDP. Sus neuronas tienen inhibicion lateral y mecanismos de adaptacion que previenen la activacion sostenida.

### SOMs con Temporal Extension
- **Chappell & Taylor (1993)**: Self-organizing maps lateralmente conectados con neuronas spiking y sinapsis leaky-integrator. Logran auto-organizacion Y segmentacion simultaneamente.

### Refractory Periods en redes spiking modernas
- **HDRP (2025)**: "Historical Dynamic Refractory Period" - Mecanismo que ajusta dinamicamente la duracion del periodo refractario basandose en el historico de actividad. Directamente relevante a tu idea de `max_active_steps`.
- **RPLIF (2025)**: "Spike-Triggered Threshold Dynamics" - Incrementa el umbral despues de cada spike, previniendo sobre-activacion.

---

## 8. Resumen de acciones recomendadas

### Prioridad 1 (fix inmediato):
- **Desactivar aprendizaje durante el gap** (`_in_gap`). Esto es el equivalente a la supresion sacadica.

### Prioridad 2 (prueba dedicada - tu propuesta):
- **Implementar `max_active_steps`** con default 5. Agregar contador de activacion consecutiva y periodo refractario.

### Prioridad 3 (mejorar convergencia):
- **Inicializacion no-ruidosa de sinapsis cercanas** (daemon) a 0.5 o pre-entrenarlas sin input.

### Prioridad 4 (investigar):
- Evaluar si la regla de aprendizaje necesita un mecanismo winner-take-all (solo el daemon con mayor match al input aprende).
- Evaluar la relacion optima entre `input_dendrite_weight` y `deamon_exc_weight`.

---

## 9. Prompt para continuar en otro chat

```
Voy a implementar "Spike Frequency Adaptation" (fatiga neuronal) en el 
Dynamic SOM de NeuroFlow. El objetivo es que las neuronas no puedan 
estar activas mas de N pasos consecutivos.

Contexto del problema:
- Los Daemons se vuelven estaticos y no se mueven
- Los pesos de las sinapsis de entrada permanecen ruidosos, no se 
  diferencian por patron
- Necesitamos un mecanismo que fuerce a las neuronas a apagarse 
  despues de max_active_steps pasos activos consecutivos

Implementacion requerida:

1. En `backend/core/brain_tensor.py` (BrainTensor):
   - Agregar parametro `max_active_steps` (int, default 5) al constructor
   - Agregar tensor `active_counts` [N] inicializado a 0
   - En `procesar()`, despues de calcular nuevos_valores:
     a) Incrementar active_counts para neuronas que estan activas
     b) Reset a 0 para neuronas que estan inactivas  
     c) Si active_counts >= max_active_steps: forzar desactivacion y 
        resetear contador
   - Proteger las NeuronaEntrada de este mecanismo (no deben fatigarse)

2. En `backend/core/constructor_tensor.py` (ConstructorTensor.compilar):
   - Pasar max_active_steps al constructor de BrainTensor
   - Si no viene en config, usar 5 como default

3. En `backend/experiments/dynamic_som.py` (DynamicSOMExperiment):
   - Agregar parametro configurable `max_active_steps` (default 5)
   - Pasarlo a ConstructorTensor.compilar
   - Ademas, FIX: desactivar learn() durante self._in_gap (supresion 
     sacadica)
   - Hacer que max_active_steps sea un "soft param" modificable en 
     caliente

4. En `backend/api/routes.py`:
   - Agregar max_active_steps al default_config del dynamic_som

5. En `frontend/src/App.tsx`:
   - Agregar max_active_steps al default_config del dynamic_som

6. En `frontend/src/components/Sidebar.tsx`:
   - Agregar control numerico para max_active_steps (rango 1-50, 
     step 1) en la seccion de configuracion del Dynamic SOM

7. En `frontend/src/types/index.ts`:
   - Agregar max_active_steps a ExperimentConfig

Notas importantes:
- max_active_steps=0 o un valor muy alto deberia desactivar 
  el mecanismo (sin fatiga)
- El parametro debe ser "soft" (modificable sin reconnect)
- Los stats deberian incluir "fatigued_cells" para monitorear 
  cuantas neuronas estan en periodo refractario en cada step
- Correr tests despues: ./venv/bin/python -m pytest backend/tests/ -x -q
```

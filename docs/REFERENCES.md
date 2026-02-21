# References

Bibliography organized by subject area. Each entry includes its relevance to NeuroFlow.

---

## Philosophy of mind and consciousness

### Dennett, D. C. (1991). *Consciousness Explained*. Little, Brown and Company.

The central work that inspires NeuroFlow's architecture. Dennett demonstrates that there is no "Cartesian theater" — no central observer where consciousness is projected — and instead proposes a **Multiple Drafts** model where parallel processes compete to emerge. He adopts Selfridge's Pandemonium model and introduces the **Joycean machine**: a virtual entity that emerges from the distributed competition of daemons.

> *"There is no single, definitive 'stream of consciousness,' because
> there is no central Headquarters, no Cartesian Theater where 'it all
> comes together'."*

### Selfridge, O. G. (1959). "Pandemonium: A Paradigm for Learning." *Symposium on the Mechanization of Thought Processes*.

The original daemon model: semi-independent processes that operate in parallel and compete with each other. It was one of the first connectionist programs, with evolutionary models where connection strengths evolve over time. NeuroFlow literally implements this dynamics in its neuronal fabric.

### Hofstadter, D. R. & Dennett, D. C. (Eds.). (1981). *The Mind's I: Fantasies and Reflections on Self and Soul*. Basic Books. ISBN 978-0-465-03091-0.

Anthology bringing together essays by Borges, Dawkins, Searle and others on artificial intelligence, consciousness, the nature of the self and the soul. This was the book that introduced NeuroFlow's author to the philosophy of mind and connected him with the work of Dawkins and Dennett.

---

## Evolutionary biology and memetics

### Dawkins, R. (1976). *The Selfish Gene*. Oxford University Press. ISBN 0-19-929114-4.

Introduces the concept of **memes** — units of cultural information that replicate and evolve by natural selection, analogous to genes. Chapter 11, "Memes: the new replicators," is foundational for memetics. The view of reality as evolutionary, iterative and distributed deeply informs NeuroFlow's philosophy.

---

## Computational neuroscience

### Hawkins, J. & Blakeslee, S. (2004). *On Intelligence*. Times Books. ISBN 978-0-8050-7456-7.

Proposes that the neocortex is a **prediction machine**: it learns a hierarchical model of the world and predicts future inputs. Founded **Numenta** to develop HTM (Hierarchical Temporal Memory), a biologically informed approach based on the physiology of pyramidal neurons in the neocortex. NeuroFlow draws heavily on the idea of hierarchical prediction and the columnar organization of the neocortex.

### Kandel, E. R. (2001). "The Molecular Biology of Memory Storage: A Dialogue Between Genes and Synapses." *Bioscience Reports*, 21(5), 507-522. Nobel Lecture.

Kandel's studies on ***Aplysia californica*** uncovered the synaptic mechanisms of learning and memory: long-term memory involves synaptic remodeling and growth of new synapses. *Aplysia* is the target model for Stage 5 (Motor Agents) due to its simple nervous system (~20,000 neurons) that allows behavioral changes to be attributed to specific neurons and synapses.

### Kandel, E. R., Schwartz, J. H. & Jessell, T. M. (2000). *Principles of Neural Science* (4th ed.). McGraw-Hill.

Comprehensive reference on neuroscience, including nociceptors, motor circuits and cortical organization.

---

## Self-organization models

### Kohonen, T. (1990). "The Self-Organizing Map." *Proceedings of the IEEE*, 78(9), 1464-1480.

The **self-organizing map** (SOM) is a neural network where cells tune to input patterns through unsupervised learning, creating spatially organized representations analogous to the brain's topographic maps. It uses competitive lateral interactions with a **Mexican hat** pattern (local excitation, lateral inhibition). NeuroFlow implements this principle with its Daemon masks.

### Hubel, D. H. & Wiesel, T. N. (1962). "Receptive Fields, Binocular Interaction and Functional Architecture in the Cat's Visual Cortex." *Journal of Physiology*, 160(1), 106-154.

Discovered the organization of receptive fields in the cat's visual cortex — the biological basis of the Mexican hat and lateral inhibition. Nobel Prize in Physiology or Medicine (1981). Their observation of nature is the biological foundation of the Daemon connections in NeuroFlow.

---

## Cellular automata and computation

### Wolfram, S. (2002). *A New Kind of Science*. Wolfram Media.

Systematic study of elementary cellular automata. Demonstrates that extremely simple rules (such as Rule 110) can perform universal computation. NeuroFlow uses Wolfram automata as a test bed: if the connectionist model can reproduce the 256 elementary rules, it validates the expressiveness of the Synapse→Dendrite→Neuron system.

---

## Motor circuits and locomotion

### Warp, E. et al. (2012). "Emergence of patterned activity in the developing zebrafish spinal cord." *Current Biology*, 22(2), 93-102.

### Svara, F. et al. (2023). "Molecular blueprints for spinal circuit modules controlling locomotor speed in zebrafish." *Nature Neuroscience*.

The **zebrafish** (*Danio rerio*) is the model for Stage 5 due to its transparent nervous system and well-characterized spinal motor circuits. Computational models of locomotor circuits in zebrafish show how molecular diversity translates into functional modules that control speed and type of movement.

---

## Pattern recognition and classification

### Fukunaga, K. (1990). *Introduction to Statistical Pattern Recognition* (2nd ed.). Academic Press.

### Duda, R. O., Hart, P. E. & Stork, D. G. (2001). *Pattern Classification* (2nd ed.). Wiley. ISBN 0-471-05669-3.

Foundational texts on classification in hyperspaces. They describe the progression of classification hypotheses: spheres → ellipses → ellipsoids → correlated ellipsoids → complex shapes. This theoretical framework informed the author's work at GIAR (UTN) on handwritten text recognition and image descriptors (Hu moments).

*Note: The author conducted this study independently and later discovered that the existing literature confirmed his findings.*

---

## Pain and nociceptors

### Melzack, R. & Wall, P. D. (1965). "Pain Mechanisms: A New Theory." *Science*, 150(3699), 971-979.

The **gate control theory of pain** proposes that pain perception is not a direct channel but a system where excitatory signals (C fibers, nociceptive) and inhibitory signals (Aβ fibers, mechanical) compete in the dorsal horn of the spinal cord. This model of competition between excitation and inhibition is directly analogous to the daemon dynamics in NeuroFlow.

---

## Convolutional networks and visualization

### Mordvintsev, A. et al. (2015). "Inceptionism: Going Deeper into Neural Networks." *Google Research Blog*.

**Deep Dream** demonstrates that a convolutional network trained to classify images contains enough information to *generate* them. Layers extract progressively more complex features — from edges to complete objects. The ability to organize images by similarity is analogous to what a SOM does, connecting modern artificial vision with NeuroFlow's connectionist approach.

---

## Science communication

### DotCSV (Carlos Santana Vega). YouTube channel.

Science communication channel on artificial intelligence in Spanish (~1M subscribers). His explanations of convolutional networks and Deep Dream illuminated the connection between similarity-based organization in deep networks and self-organizing maps.

URL: [youtube.com/@DotCSV](https://www.youtube.com/@DotCSV)

---

← Back to [README](../README.md)

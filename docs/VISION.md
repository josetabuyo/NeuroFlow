# Vision and Philosophy

> *"There is no single, definitive 'stream of consciousness,' because there
> is no central Headquarters, no Cartesian Theater where 'it all comes
> together'."*
> — Daniel Dennett, *Consciousness Explained* (1991)

---

## What does NeuroFlow seek?

NeuroFlow seeks to build a **model of the mind** — not just of language
(which LLMs already address successfully), but of the cognitive
capabilities that remain largely unexplored:

- **Movement**: motor control, muscular coordination, locomotion
- **Visual perception**: image recognition, topographic organization
- **Depth of reasoning**: what we call *intuition* — multiple levels of
  abstraction and relationships that a single layer of explanation cannot
  capture

The model is compatible with language, but the priority is to tackle what
has not yet been conquered.

---

## The Daemon: the fundamental unit

### No Cartesian theater

The usual fantasy is that there exists a central observer — a homunculus
sitting in a "Cartesian theater" — watching a screen where consciousness
is projected. Daniel Dennett demonstrates in *Consciousness Explained*
(1991) that this image is an illusion.

Instead, Dennett adopts Oliver Selfridge's **Pandemonium** model (1959): a
multitude of semi-independent processes — **daemons** — that operate in
parallel. When a problem arises, the daemons compete among themselves,
shouting "Me! Me! Let me handle it!". One wins the competition and
tackles the problem; if it fails, others take over.

From this distributed competition emerges what Dennett calls the
**Joycean machine**: a "virtual captain" that creates the *illusion* of
a unified self and executive control. But there is no daemon that rules
permanently — it is shifting coalitions that produce order, through what
Dennett describes as "a kind of internal political miracle".

### The daemon in NeuroFlow

NeuroFlow seeks the daemon through a **purely connectionist network**:

1. There is no centralized processing
2. There is no server or observer
3. Concepts compete to emerge
4. Stability arises from local rules

In our 2D tissue of neurons, daemons are **activation bubbles** that:

- Move (up, down, left, right)
- Are stable (they do not dissipate)
- Resist noise
- Compete by exclusion (like musical notes that do not tolerate dissonance)
- Self-balance (~50% of neurons active at all times)
- Converge to a new state when manipulated externally

---

## The Mexican hat: biological inspiration

Teuvo Kohonen, when developing **Self-Organizing Maps** (SOMs), drew
inspiration from **Hubel and Wiesel's** observations of the cat's visual
cortex — work that earned them the Nobel Prize (1981). What they
discovered is a lateral connection pattern that we now call the
**Mexican hat** (Mexican hat / difference of Gaussians):

```
         Excitation                        Inhibition
      ┌─────────────┐                 ┌───────────────────┐
      │  Center:    │                 │  Surround:        │
      │  nearby     │    Silence      │  distant          │
      │  neighbors  │    (gap)        │  neighbors        │
      │  excite     │                 │  inhibit          │
      └─────────────┘                 └───────────────────┘

      Dendrite weight > 0             Dendrite weight < 0
```

This observation **comes straight from nature**: the brain uses local
excitation with lateral inhibition to create orderly topographic maps.
NeuroFlow replicates this principle with its Daemon mask system
(`E G I DE DI`).

---

## Prediction: the brain as a predictive machine

Jeff Hawkins, in *On Intelligence* (2004), proposes that the neocortex is
fundamentally a **prediction machine**: it learns a model of the world
and uses it to predict future inputs, with hierarchical regions that
predict their own input sequences.

NeuroFlow incorporates this vision: the perception of "guessing the
future" is part of the model's scope. Through Wolfram automaton-like
dynamics, we seek stable connections that can **accompany or reflect**
what occurs with muscular movements — an embodied predictive model.

---

## Distributed and evolutionary vision

NeuroFlow's vision is framed within a modern understanding of reality
as something **evolutionary, iterative, and distributed**:

- Just as Darwin showed that species are not designed by a central
  creator but emerge through distributed natural selection...
- Just as the economy is not directed by a central genius but emerges
  from the distributed work of millions of agents...
- So too neuronal activity is not directed by an observer in a theater,
  but emerges from the distributed competition of local processes.

NeuroFlow's daemons are the computational expression of this vision.

---

## Parallel with musical notes

An unexpected finding: daemons behave analogously to musical notes. Just
as we cannot tolerate three simultaneous tones without dissonance — an
"uncomfortable key" that we perceive strongly — daemons exclude each
other when competing for the same space.

This opens a **future hypothesis**: a neuronal model whose output is
musical notes, that synthesizes and generates music from this competitive
exclusion dynamics. It is not the current goal, but remains as a line of
research.

---

## Convolutional networks and image generation

Convolutional neural networks, such as those Google explored with
**Deep Dream** (2015), demonstrate that a network trained to classify
images contains enough information to *generate* them. The layers extract
progressively more complex features — from simple edges to complete
objects.

This capacity to organize images by similarity is analogous to what a
SOM does. NeuroFlow explores this connection: understanding which filter
to excite, which image feature to seek, opens a path toward image
generation from our connectionist model.

---

## Essential readings

To delve deeper into the ideas that inspire NeuroFlow, see
**[References](REFERENCES.md)**.

← Back to [README](../README.md)

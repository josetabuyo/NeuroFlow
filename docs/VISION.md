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

## The Mexican hat: what biology observed vs. what we build

The **Mexican hat** (difference of Gaussians) is a well-known profile in
neuroscience: center excitation surrounded by lateral inhibition. But it
is important to distinguish **what was observed** from **what was wired**.

**Kuffler (1953)** discovered center-surround receptive fields in retinal
ganglion cells — the clearest biological Mexican hat. **Hubel and Wiesel
(1962)**, building on Kuffler's work, recorded the **response profiles**
of individual neurons in the cat's visual cortex (Nobel Prize, 1981).
They found that cortical neurons respond to oriented edges, not spots of
light, with excitatory and inhibitory regions in their receptive fields.
Crucially, what they measured was the **neuron's response** — how it
fires — not the physical wiring between neurons.

**Kohonen**, when developing **Self-Organizing Maps** (SOMs), drew
inspiration from the general principle of **lateral inhibition** in
cortical topographic maps (tonotopic, somatotopic). His Mexican hat
function describes the *effect* of lateral interactions — nearby neurons
excite, distant neurons inhibit — not a specific wiring diagram.

### What NeuroFlow does differently

In NeuroFlow, the Daemon connection masks (`E G I DE DI`) define
**wiring patterns** — which neurons connect to which, and with what
polarity. These masks are not necessarily Mexican hat-shaped:

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

But when these connections operate — when the network runs — the
**emergent daemon** (the stable activation bubble) exhibits a surface
tension that *coincides* with the Mexican hat profile: an active center
that sustains itself, surrounded by a silenced zone that excludes
competing activity.

This mirrors the biological observation: what Hubel, Wiesel, and Kuffler
measured was the **response** (the effect, the surface tension), not the
connection pattern. And that is exactly what happens in NeuroFlow — the
masks define the wiring, the daemon *is* the Mexican hat.

---

## Prediction: seeing the future through the tissue

Jeff Hawkins, in *On Intelligence* (2004), proposes that the neocortex is
fundamentally a **prediction machine**: it learns a model of the world
and uses it to predict future inputs, with hierarchical regions that
predict their own input sequences.

NeuroFlow incorporates this vision, but through a specific connectionist
mechanism built on three levels of connectivity:

### Level 1 — Local connections (current stage)

The Daemon masks (`E G I DE DI`) define nearby excitatory and inhibitory
dendrites. These local connections produce the daemon: the stable
activation bubble with its characteristic surface tension. Reality
propagates through the tissue via these connections — activity spreads,
competes, and self-organizes locally.

### Level 2 — Distant input connections (SOM stage)

In the next stages, neurons will gain **distant dendrites** that read
from sensors or any external input source. These can be thought of as
long-range dendrites — the tissue's connection to reality. With these,
the tissue can learn to represent input patterns: a self-organizing map
where the spatial arrangement of daemons reflects the structure of the
input space.

### Level 3 — Extended local connections without input (prediction)

A region of the tissue with **no direct input connections**, but with
local dendrites of a **larger radius** than the standard daemon. This
region only observes the activity of the surrounding tissue — which
*is* connected to inputs.

The prediction mechanism emerges from this arrangement:

- The tissue connected to sensors reflects reality through its
  activation patterns.
- Activity propagates through the tissue with a temporal dynamic —
  patterns build, shift, and evolve over time.
- The prediction region, observing this activity through its extended
  dendrites, develops a **surface tension that reflects the probability
  of an event occurring**.
- As precursor patterns accumulate in the input-connected tissue, the
  tension rises in the prediction region.
- When the event materializes, the neuron that previously learned that
  pattern activates.

**Prediction, then, is not an explicit circuit but an emergent effect**:
seeing the future by analyzing present activity in the tissue, which in
turn is correlated with time through its sensor connections.

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

## Parallels: where competitive exclusion appears in nature

The daemon's defining behavior — mutual exclusion, where one activation
suppresses its neighbors — is not unique to our model. The same dynamic
appears in at least two natural systems, suggesting that NeuroFlow's
connectionist principle may be more fundamental than it seems.

### Musical notes and dissonance

Daemons behave analogously to musical notes. Just as we cannot tolerate
three simultaneous tones without dissonance — an "uncomfortable key"
that we perceive strongly — daemons exclude each other when competing
for the same space.

This opens a **future hypothesis**: a neuronal model whose output is
musical notes, that synthesizes and generates music from this competitive
exclusion dynamics. It is not the current goal, but remains as a line of
research.

### Antagonistic muscles and nociceptors

Every muscle in the body has an **antagonist** — a muscle that performs
the opposite movement. The relationship between them is always one of
exclusion: when one contracts, the other relaxes. This is not optional;
it is how motor control works.

When something causes pain, **nociceptors** trigger an inversion of
activity in the affected region: the muscles that were active swap roles
with their antagonists. The limb reverses direction. If you burn your
backside while walking backwards, all your muscles — starting from those
closest to the stimulus — invert their activation with their
antagonists, reversing the movement to escape the source of pain. The
same happens if you bite your tongue: the jaw muscles immediately
invert.

This is a **total parallel** with the daemon model: a region of
mutually exclusive activity where inversion propagates outward from a
stimulus. The motor cortex that controls these antagonistic muscle pairs
could work like our tissue — a surface of competing daemons where
flipping one daemon's state cascades through its neighbors, producing
coordinated movement reversal.

---

## Hierarchy without layers: a single tissue

Convolutional neural networks achieve hierarchical processing by
stacking discrete layers — each layer extracts progressively more
complex features, from edges to shapes to complete objects. Google's
**Deep Dream** (2015) demonstrated that such networks contain enough
information to *generate* images, not just classify them.

NeuroFlow does not replicate this architecture — it is not the same
thing. But the expectation is that **hierarchical behavior can emerge
spontaneously from a single tissue** with the right connectivity.

The key insight: with dendrites slightly longer than those that shape the
daemon, a signal flowing through the tissue can undergo progressive
compression and classification along its path. In one region, raw input
activates many daemons; as the signal propagates through connections
that are biased in a direction, fewer and more selective daemons respond
— a natural hierarchy arising from the flow of activity, not from
stacked layers.

This is possible because lateral connections are flexible:

- They can be **biased directionally** — dendrites that only reach
  leftward, or upward, creating a preferred flow direction.
- They can vary in **radius and density** — tighter connections in one
  region, wider in another, producing different levels of abstraction
  along the tissue.
- They can be **asymmetric** — more excitation in one direction, more
  inhibition in another.

No composition of separate networks is needed. A single tissue with
spatially varying connectivity can express the same progressive
abstraction that convolutional networks achieve with explicit layers.
Whether the daemons are flowing or stationary, the hierarchy is a
property of how the signal travels through the tissue — not of how
many networks are stacked.

This also connects to the SOM: the capacity to organize inputs by
similarity is analogous to what a self-organizing map does, but here
the organization can have *depth* — not just spatial arrangement, but
hierarchical compression along the axis of signal flow.

---

## Essential readings

To delve deeper into the ideas that inspire NeuroFlow, see
**[References](REFERENCES.md)**.

← Back to [README](../README.md)

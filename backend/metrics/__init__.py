"""Visualization-only metrics (daemon detection, stability, ...).

Nothing here feeds back into the simulation — it only observes a snapshot of
the network to produce HUD data. Kept out of core/ (the simulation engine)
and decoupled from the play loop's frame rate: callers choose their own
cadence for calling into this package.
"""

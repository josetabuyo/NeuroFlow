"""Base class for experiment handlers."""

from __future__ import annotations


class ExperimentHandler:
    """Base class for experiment-specific auxiliary logic.

    Handlers centralise methods that the config references by name via
    ``handler_method``.  Region-level ``handler_method`` binds to a method
    that returns ``list[list[str]]`` (neuron labels).  Source-level
    ``handler_method`` can override injection logic for nociceptors and
    other source regions.
    """

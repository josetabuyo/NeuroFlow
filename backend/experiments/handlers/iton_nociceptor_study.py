"""Handler for the ITON Nociceptor Study experiment."""

from __future__ import annotations

from .base import ExperimentHandler


class ITON_Nociceptor_Study(ExperimentHandler):
    """Auxiliary methods for the ITON two-class nociceptor experiment.

    Region handler_method bindings:
      label_L  → neuron labels for output_L  (returns [["L"]])
      label_T  → neuron labels for output_T  (returns [["T"]])
    """

    def label_L(self) -> list[list[str]]:
        return [["L"]]

    def label_T(self) -> list[list[str]]:
        return [["T"]]

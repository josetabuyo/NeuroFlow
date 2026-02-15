"""Neurona y NeuronaEntrada — unidades de cómputo del modelo neuronal.

Neurona:
  procesar() → fuzzy OR: max(dendritas) + min(dendritas) → setTension
  activar()  → tension > umbral ? valor=1 : valor=0

NeuronaEntrada:
  procesar() → no-op
  activar()  → no-op
  Solo cambia vía activar_external(valor)
"""

from __future__ import annotations

from .dendrita import Dendrita


class Neurona:
    """Unidad de cómputo: evalúa dendritas con fuzzy OR competitivo."""

    __slots__ = ("id", "valor", "tension_superficial", "dendritas", "umbral")

    def __init__(
        self,
        id: str,
        dendritas: list[Dendrita] | None = None,
        umbral: float = 0.0,
    ) -> None:
        self.id = id
        self.valor: float = 0.0
        self.tension_superficial: float = 0.0
        self.dendritas: list[Dendrita] = dendritas if dendritas is not None else []
        self.umbral = umbral

    def _set_tension(self, tension: float) -> None:
        """Normaliza y setea la tensión superficial en [-1, 1]."""
        if tension < -1.0:
            self.tension_superficial = -1.0
        elif tension > 1.0:
            self.tension_superficial = 1.0
        else:
            self.tension_superficial = tension

    def procesar(self) -> None:
        """Fuzzy OR competitivo: max(dendritas) + min(dendritas) → tensión."""
        if not self.dendritas:
            self._set_tension(0.0)
            return

        max_valor = 0.0
        min_valor = 0.0

        for dendrita in self.dendritas:
            dendrita.procesar()
            if dendrita.valor > max_valor:
                max_valor = dendrita.valor
            if dendrita.valor < min_valor:
                min_valor = dendrita.valor

        valor = max_valor + min_valor
        self._set_tension(valor)

    def activar(self) -> None:
        """Aplica umbral sobre tensión: si tension > umbral → valor=1, sino valor=0."""
        if self.tension_superficial > self.umbral:
            self.valor = 1.0
        else:
            self.valor = 0.0

    def __repr__(self) -> str:
        return f"Neurona(id={self.id}, valor={self.valor})"


class NeuronaEntrada(Neurona):
    """Neurona de entrada: sin dendritas, valor seteado externamente.

    La Red la procesa igual que las demás, pero internamente no hace nada.
    """

    def __init__(self, id: str) -> None:
        super().__init__(id=id, dendritas=[], umbral=0.0)

    def procesar(self) -> None:
        """No-op: la neurona de entrada no tiene dendritas que procesar."""
        pass

    def activar(self) -> None:
        """No-op: el valor de la neurona de entrada se setea externamente."""
        pass

    def activar_external(self, valor: float) -> None:
        """Setea el valor de la neurona de entrada desde el exterior."""
        self.tension_superficial = valor
        self.valor = valor

    def __repr__(self) -> str:
        return f"NeuronaEntrada(id={self.id}, valor={self.valor})"

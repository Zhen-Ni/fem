#!/usr/bin/env python3


from typing import Optional, SupportsFloat

from .common import Readonly

__all__ = ['Material', 'STEEL', 'ALUMINIUM']


class Material(Readonly):
    """Class for defining a material.

    Parameters
    ----------
    E : float
        The Young's modulus.
    nu : float
        The Poisson's ratio.
    rho : float, optional
        Density.
    name : string, optional
        Name of the material.
    """

    def __init__(self,
                 E: SupportsFloat,
                 nu: SupportsFloat,
                 rho: Optional[SupportsFloat] = None,
                 name: Optional[str] = None):
        super().__init__()
        self._E = float(E)
        self._nu = float(nu)
        self._rho = None if rho is None else float(rho)
        self._G = self._E / (2 * (1 + self._nu))
        self._name = 'unnamed material' if name is None else name

    def __repr__(self):
        return "Material: {name}".format(name=self.name)

    @property
    def E(self) -> float:
        return self._E

    @property
    def nu(self) -> float:
        return self._nu

    @property
    def rho(self) -> Optional[float]:
        return self._rho

    @property
    def G(self) -> float:
        return self._G

    @property
    def name(self) -> str:
        return self._name


STEEL = Material(210e9, 0.3, 7800, 'Steel')
ALUMINIUM = Material(70e9, 0.3, 2710, 'Aluminium')

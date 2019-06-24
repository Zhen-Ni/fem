#!/usr/bin/env python3

from .readonly import Readonly

__all__ = ['Material', 'STEEL']

class Material(Readonly):
    """Class for defining a material.
    
    Parameters
    ----------
    E: float
        The Young's modulus.
    nu: float
        The Poisson's ratio.
    rho: float
        Density.
    name: string
        Name of the material.
    """
    def __init__(self, E=None, nu=None, rho=None, name=None):
        super().__init__()
        self.E = E
        self.nu = nu
        self.rho = rho
        self.G = E / (2*(1+nu))
        self.name = name
        self._set_readonly()

    def __repr__(self):
        return "Material: {name}".format(name=self.name)


STEEL = Material(210e9,0.3,7800,'Steel')

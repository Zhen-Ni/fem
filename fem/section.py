#!/usr/bin/env python3

from math import pi
import numpy as np
from typing import Optional

from .common import Readonly
from .material import Material


__all__ = ['SolidSection', 'ShellSection', 'BeamSection']


class Section(Readonly):
    """Base class of section.

    Parameters
    ----------
    material : Material object
        Material used for this section.
    name : string, optional
        Name of this section.
    """

    def __init__(self, material: Material, name: Optional[str] = None):
        super().__init__()
        self._material = material
        self._name = 'unnamed section' if name is None else name

    def __repr__(self):
        return "Section: {name}".format(name=self.name)

    @property
    def material(self) -> Material:
        return self._material

    @property
    def name(self) -> str:
        return self._name


class SolidSection(Section):
    """Class of solid section.

    Parameters
    ----------
    material : Material object
        Material used for this section.
    name : string, optional
        Name of this section.

    See Also
    --------
    ShellSection
    BeamSection
    """

    def __init__(self,
                 material: Material,
                 name: Optional[str] = None):
        super().__init__(material, name)

    def __repr__(self):
        return "SolidSection: {name}".format(name=self.name)


class ShellSection(Section):
    """Class of shell section.

    Parameters
    ----------
    material: Material object
        Material used for this section.
    h : float
        Thickness of the shell.
    name : string, optional
        Name of this section.

    See Also
    --------
    SolidSection
    BeamSection
    """

    def __init__(self,
                 material: Material,
                 h: float,
                 name: Optional[str] = None):
        super().__init__(material, name)
        self._h = h
        # Shell correction factor is default to 5 / 6 for isotropic
        # material.
        self._kappa = 5 / 6

    def __repr__(self):
        return "ShellSection: {name}".format(name=self.name)

    @property
    def h(self) -> float:
        return self._h

    @property
    def kappa(self) -> float:
        return self._kappa


class BeamSection(Section):
    """Class of shell section.

    The beam is modelled along the x-axis.

    Parameters
    ----------
    material: Material object
        Material used for this section.
    A : float
        Cross sectional area of the beam.
    Ix : float
        Moment of inertia in the x direction.
    Iy : float
        Moment of inertia in the y direction.
    Iz : float
        Moment of inertia in the z direction.
    name : string, optional
        Name of this section.
    kappa : float, optional
        Shear correction factor for FSDT.
    torsion_coefficient : float, optional
        Correction factor for torsion.

    See Also
    --------
    SolidSection
    BeamSection
    """

    def __init__(self,
                 material: Material,
                 A: float,
                 Ix: float,
                 Iy: float,
                 Iz: float,
                 name: Optional[str] = None,
                 *,
                 kappa: Optional[float] = None,
                 torsion_coefficient: Optional[float] = None):
        super().__init__(material, name)
        self._A = A
        self._Ix = Ix
        self._Iy = Iy
        self._Iz = Iz
        if kappa is None:
            kappa = 1
        if torsion_coefficient is None:
            torsion_coefficient = 1
        self._kappa = kappa
        self._torsion_coefficient = torsion_coefficient

    def __repr__(self):
        return "BeamSection: {name}".format(name=self.name)

    @property
    def A(self) -> float:
        return self._A

    @property
    def Ix(self) -> float:
        return self._Ix

    @property
    def Iy(self) -> float:
        return self._Iy

    @property
    def Iz(self) -> float:
        return self._Iz

    @property
    def kappa(self) -> float:
        return self._kappa

    @property
    def torsion_coefficient(self) -> float:
        return self._torsion_coefficient

    @staticmethod
    def circular(material: Material, r: float, name: Optional[str] = None):
        """Create a instance of SectionBeam for circular.

        Parameters
        ----------
        material: Material object
            Material used for this section.
        r : float
            Radius of the cross section.
        name : string, optional
            Name of this section.
        """
        A = pi * r ** 2
        Ix = 0.5 * pi * r ** 4
        Iy = 0.5 * Ix
        Iz = Iy
        nu = material.nu
        kappa = 6 * (1 + nu) / (7 + 6 * nu)
        return BeamSection(material, A, Ix, Iy, Iz, name=name, kappa=kappa)

    @staticmethod
    def pipe(material: Material,
             R: float,
             r: float,
             name: Optional[str] = None):
        """Create a instance of SectionBeam for pipe.

        Parameters
        ----------
        material : Material object
            Material used for this section.
        R : float
            Radius of the outer circle of the pipe.
        r : float
            Radius of the inner circle of the pipe.
        name : string, optional
            Name of this section.
        """
        A = pi * (R ** 2 - r ** 2)
        Ix = 0.5 * pi * (R ** 4 - r ** 4)
        Iy = 0.5 * Ix
        Iz = Iy
        nu = material.nu
        m = R / r
        kappa = 6 * (1 + nu) * (1 + m ** 2) ** 2
        kappa /= (7 + 6 * nu) * (1 + m ** 2) ** 2 + (20 + 12 * nu) * m ** 2
        return BeamSection(material, A, Ix, Iy, Iz, name=name, kappa=kappa)

    @staticmethod
    def rectangular(material: Material,
                    width: float,
                    height: float,
                    name: Optional[str] = None):
        """Create a instance of SectionBeam for rectangular.

        Parameters
        ----------
        material : Material object
            Material used for this section.
        width : float
            Width of the rectangular.
        height : float
            Height of the rectangular.
        name : string, optional
            Name of this section.

        Notes
        -----
        An approximation is used for Ix (or J) in this function. Maybe a better
        solution can be obtained by the Prandtl stress function approach.
        """
        a, b = width, height
        A = a * b
        Iy = a * b ** 3 / 12
        Iz = a ** 3 * b / 12
        Ix = Iy + Iz
        nu = material.nu
        kappa = 10 * (1 + nu)/(12 + 11 * nu)
        if np.log(max(a, b)/min(a, b)) >= 100:
            beta = 0.333
        else:
            from scipy.interpolate import interp1d
            ratios = np.log(
                np.array([1., 1.2, 1.5, 1.75, 2, 2.5, 3,  10, 100.]))
            betas = np.array([0.141, 0.166, 0.196, 0.214, 0.229, 0.249, 0.263,
                              0.313, 0.333])
            beta = interp1d(ratios, betas, kind='cubic')(
                np.log(max(a, b) / min(a, b)))
        torsion_coefficient = beta * min(a, b) ** 3 * max(a, b) / Ix

        return BeamSection(material, A, Ix, Iy, Iz, name=name, kappa=kappa,
                           torsion_coefficient=torsion_coefficient)


if __name__ == '__main__':
    from material import STEEL
    section_raft = ShellSection(STEEL, 0.01, 'Raft')

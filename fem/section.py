#!/usr/bin/env python3

from math import pi
import numpy as np

from .readonly import Readonly


__all__ = ['SectionShell', 'SectionBeam']

class Section(Readonly):
    """Base class of section.
    
    Parameters
    ----------
    material: Material object
        Material used for this section. 
    name: string, optional
        Name of this section.
    """
    
    def __init__(self, material, name):
        super().__init__()
        self.material = material
        self.name = name

    def __repr__(self):
        return "Section: {name}".format(name=self.name)

    def __getattr__(self, name):
        try:
            res = getattr(self.material, name)
        except AttributeError:
            raise AttributeError("'Section' object has no"
                                 " attribute '{name}'".format(name=name))
        return res

class SectionShell(Section):
    """Class of shell section.
    
    Parameters
    ----------
    material: Material object
        Material used for this section. 
    h: float
        Thickness of the shell.
    name: string, optional
        Name of this section.
    
    See Also
    --------
    SectionBeam
    """
    def __init__(self, material,h, name=None):
        super().__init__(material, name)
        self.h = h
        self.kappa = 5 / 6
        self._set_readonly()

    def __repr__(self):
        return "Section: {name}".format(name=self.name)


class SectionBeam(Section):
    """Class of shell section.
    
    Parameters
    ----------
    material: Material object
        Material used for this section. 
    A: float
        Cross sectional area of the beam.
    Ix: float
        Moment of inertia in the x direction.
    Iy: float
        Moment of inertia in the y direction.
    Iz: float
        Moment of inertia in the z direction.
    name: string, optional
        Name of this section.
    kappa: float, optional
        Shear correction factor for FSDT.
    torsion_coefficient: float, optional
        Correction factor for torsion.
    
    See Also
    --------
    SectionBeam
    """
    def __init__(self, material,A,Ix,Iy,Iz, name=None, kappa=None, torsion_coefficient=None):
        super().__init__(material, name)
        self.A = A
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        if kappa is None:
            kappa = 1
        if torsion_coefficient is None:
            torsion_coefficient = 1
        self.kappa = kappa
        self.torsion_coefficient = torsion_coefficient
        self._set_readonly()

    def __repr__(self):
        return "Section: {name}".format(name=self.name)

    @staticmethod
    def circular(material, r, name=None):
        """Create a instance of SectionBeam for circular.
        
        Parameters
        ----------
        material: Material object
            Material used for this section. 
        r: float
            Radius of the cross section.
        name: string, optional
            Name of this section.
        """
        A = pi * r ** 2
        Ix = 0.5 * pi * r ** 4
        Iy = 0.5 * Ix
        Iz = Iy
        nu = material.nu
        kappa = 6*(1+nu)/(7+6*nu)
        return SectionBeam(material, A, Ix, Iy, Iz,name=name, kappa=kappa)

    @staticmethod
    def pipe(material,R, r, name=None):
        """Create a instance of SectionBeam for pipe.
        
        Parameters
        ----------
        material: Material object
            Material used for this section. 
        R: float
            Radius of the outer circle of the pipe.
        r: float
            Radius of the inner circle of the pipe.
        name: string, optional
            Name of this section.
        """
        A = pi * (R ** 2 - r ** 2)
        Ix = 0.5 * pi * (R ** 4 - r ** 4)
        Iy = 0.5 * Ix
        Iz = Iy
        nu = material.nu
        m = R / r
        kappa = 6*(1+nu)*(1+m**2)**2
        kappa /= (7+6*nu)*(1+m**2)**2+(20+12*nu)*m**2
        return SectionBeam(material, A, Ix, Iy, Iz,name=name, kappa=kappa)

    @staticmethod
    def rectangular(material, a, b, name=None):
        """Create a instance of SectionBeam for rectangular.
        
        Parameters
        ----------
        material: Material object
            Material used for this section. 
        a: float
            Length of the rectangular.
        b: float
            Width of the rectangular.
        name: string, optional
            Name of this section.
        
        Notes
        -----
        An approximation is used for Ix (or J) in this function. Maybe a better
        solution can be obtained by the Prandtl stress function approach.
        """
        A = a*b
        Iy = a*b**3/12
        Iz = a**3*b/12
        Ix = Iy + Iz
        nu = material.nu
        kappa = 10*(1+nu)/(12+11*nu)
        if np.log(max(a,b)/min(a,b)) >= 100:
            beta = 0.333
        else:
            from scipy.interpolate import interp1d
            ratios = np.log(np.array([  1.,1.2,1.5,1.75, 2,2.5 , 3,  10,100.  ]))
            betas = np.array([0.141, 0.166, 0.196, 0.214, 0.229, 0.249, 0.263,
                              0.313, 0.333])
            beta = interp1d(ratios, betas, kind='cubic')(np.log(max(a,b)/min(a,b)))
        torsion_coefficient = beta * min(a,b)**3*max(a,b) / Ix

        return SectionBeam(material, A, Ix, Iy, Iz,name=name, kappa = kappa,
                           torsion_coefficient=torsion_coefficient)

if __name__ == '__main__':
    from material import STEEL
    section_raft = SectionShell(STEEL,0.01,'Raft')

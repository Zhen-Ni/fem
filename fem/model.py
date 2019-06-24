#!/usr/bin/env python3


import numpy as np
from numpy.linalg import norm


class Model:
    """Create a model for finite element analysis.
    
    Forces may be added to Model objects.
    
    Parameters
    ----------
    assembly: Assembly object
        The assembly to be analyzed.    
    """
    def __init__(self, assembly):
        self._assembly = assembly
        self._forces = []
        self._forces_gravity = []
        self._force_vector = None
        self._is_initialized = False

    @property
    def nodes(self):
        return self._assembly.nodes

    @property
    def assembly(self):
        return self._assembly

    def _uninitialize(self):
        self._is_initialized = False
        self._force_vector = None

    def _initialize(self):
        if not self._is_initialized:
            self._do_initialize()

    def _do_initialize(self):
        self._force_vector = self._get_force_vector()
        self._is_initialized = True

    def add_force(self, f, node, dof):
        """Add force to the model.
        
        Parameters
        ----------
        f: float
            Force.
        node: int
            The index of the node.
        dof: int
            The degree of freedom of the node.
        """
        self._uninitialize()
        self._forces.append((f, node*6+dof))
        return self

    def add_gravity(self, g, direction=(0,0,-1)):
        """Add gravity to the model.
        
        Parameters
        ----------
        g: float
            Gravity.
        direction: iterable
            The direction of the gravity.
        """
        self._uninitialize()
        direction = np.array(direction)
        direction = direction / norm(direction)
        self._forces_gravity.append((g,direction))


    @property
    def forces(self):
        return self._forces

    @property
    def gravities(self):
        return self._forces_gravity

    def remove_force(self, idx=None):
        """Remove force from the assembly.
        
        Parameters
        ----------
        idx: int, optional
            Index of the force to be removed. If `idx` is None, all the forces
            will be removed. (default to None)
        """
        self._uninitialize()
        if idx is None:
            self._forces = []
        else:
            self._forces.pop(idx)
        return self

    def remove_gravity(self, idx=None):
        """Remove gravity from the assembly.
        
        Parameters
        ----------
        idx: int, optional
            Index of the gravity to be removed. If `idx` is None, all the
            gravities will be removed. (default to None)
        """
        self._uninitialize()
        if idx is None:
            self._forces_gravity = []
        else:
            self._forces_gravity.pop(idx)
        return self

    def _get_force_vector(self):
        F = np.zeros(self._assembly.K.shape[0])
        for f, dof in self._forces:
            F[dof] += f
        mass_component = self._assembly.M.diagonal()
        for g, direction in self._forces_gravity:
            gx, gy, gz = g * direction
            for i in range(len(F)//6):
                F[i*6] += gx * mass_component[i*6]
                F[i*6+1] += gy * mass_component[i*6+1]
                F[i*6+2] += gz * mass_component[i*6+2]
        return F

    @property
    def M(self):
        return self._assembly.M

    @property
    def K(self):
        return self._assembly.K

    @property
    def C(self):
        return self._assembly.C

    @property
    def F(self):
        self._initialize()
        return self._force_vector

    @property
    def to_dataset(self):
        return self.assembly.to_dataset


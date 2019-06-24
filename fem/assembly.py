#!/usr/bin/env python3


import numpy as np

from .dataset import *

class Assembly:
    """Assemble the parts into an assembly.
    
    Each part of the assembly should have the same node list, whether they have
    been used in their element list.
    
    Parameters
    ----------
    parts: iterable
        Parts in the assembly. The parts should have the same node list.
    """
    def __init__(self, parts):
        if not np.iterable(parts):
            parts = [parts]

        nodes = parts[0].nodes
        for p in parts[1:]:
            if not (nodes == p.nodes).all():
                raise ValueError('nodes array should be the same in each part')
        self._nodes = nodes
        self._parts = parts
        self._stiffness_matrix = None
        self._mass_matrix = None
        self._damping_matrix = None
        self._springs = []
        self._masses = []
        self._is_initialized = False

    @property
    def parts(self):
        return self._parts

    @property
    def nodes(self):
        return self._nodes

    def add_spring(self, k, node1, dof1, node2=None, dof2=None):
        """Add a spring to the assembly.
        
        Users may define a spring connecting two nodes or connecting a node to
        the ground.
        
        Parameters
        ----------
        k: float
            Stiffness of the spring.
        node1: int
            Index of the first node.
        dof1: int
            The degree of freedom of the first node.
        node2: int, optional
            Index of the second node. If node2 is None, the spring is connected
            to the ground. (default to None)
        dof2: int, optional
            The degree of freedom of the second node. If dof2 is None, dof2 is
            the same as dof1. (default to None)
        """
        self._uninitialize()
        if node2 is None:
            self._springs.append((k, node1*6+dof1, None))
        else:
            if dof2 is None:
                dof2 = dof1
            self._springs.append((k, node1*6+dof1, node2*6+dof2))
        return self

    @property
    def springs(self):
        return self._springs

    def remove_spring(self, idx=None):
        """Remove a spring or springs from the assembly.
        
        Parameters
        ----------
        idx: int, optional
            Index of the spring to be removed. If `idx` is None, the all 
            springs will be removed. (default to None)
        """
        self._uninitialize()
        if idx is None:
            self._springs = []
        else:
            self._springs.pop(idx)
        return self

    def add_mass(self, m, node, dof):
        """Add a mass point to the assembly.
        
        Parameters
        ----------
        m: float
            The mass.
        node: int
            The index of the node.
        dof: int
            The degree of freedom of the node.
        """
        self._uninitialize()
        self._masses.append((m, node*6+dof))
        return self

    @property
    def masses(self):
        return self._masses

    def remove_mass(self, idx=None):
        """Remove mass point from the assembly.
        
        Parameters
        ----------
        idx: int, optional
            Index of the mass to be removed. If `idx` is None, all the mass
            points will be removed. (default to None)
        """
        self._uninitialize()
        if idx is None:
            self._masses = []
        else:
            self._masses.pop(idx)
        return self

    def _uninitialize(self):
        self._is_initialized = False
        self._stiffness_matrix = None
        self._mass_matrix = None
        self._damping_matrix = None

    def _initialize(self):
        if not self._is_initialized:
            self._do_initialize()

    def _do_initialize(self):
        self._stiffness_matrix = self._get_stiffness_matrix()
        self._mass_matrix = self._get_mass_matrix()
        self._damping_matrix = self._get_damping_matrix()
        self._is_initialized = True

    def _get_stiffness_matrix(self):
        K = self._parts[0].K.tocsc().copy()
        for p in self._parts[1:]:
            K += p.K.tocsc()
        for k, dof1, dof2 in self._springs:
            if dof2 is None:
                K[dof1, dof1] += k
            else:
                K[dof1, dof1] += k
                K[dof1, dof2] += -k
                K[dof2, dof1] += -k
                K[dof2, dof2] += k
        return K

    def _get_mass_matrix(self):
        M = self._parts[0].M.tocsc().copy()
        for p in self._parts[1:]:
            M += p.M.tocsc()
        for m, dof in self._masses:
            M[dof, dof] += m
        return M

    def _get_damping_matrix(self):
        C = self._parts[0].C.tocsc().copy()
        for p in self._parts[1:]:
            C += p.C.tocsc()
        return C

    @property
    def K(self):
        self._initialize()
        return self._stiffness_matrix

    @property
    def M(self):
        self._initialize()
        return self._mass_matrix

    @property
    def C(self):
        self._initialize()
        return self._damping_matrix

    def to_dataset(self, which='all'):
        """Write assembly information to dataset.
        
        Parameters
        ----------
        which: 'all' | 'mesh' | 'spring' | 'mass'
            Define which part of the assembly to export.
        
        Returns
        -------
        ds: DataSet
            A DataSet object containing geometry information of the assembly.
        """
        which = which.lower()
        points = [Point(i) for i in self._nodes]
        ds = DataSet(points)
        cells = []
        if which == 'all':
            self._to_dataset_helper_mesh(cells)
            self._to_dataset_helper_spring(cells)
            self._to_dataset_helper_mass(cells)
        elif which == 'mesh':
            self._to_dataset_helper_mesh(cells)
        elif which == 'spring':
            self._to_dataset_helper_spring(cells)
        elif which == 'mass':
            self._to_dataset_helper_mass(cells)
        else:
            raise AttributeError("'which' should be in 'all' | 'mesh' | 'sprin"
                                 "g' | 'mass'")
        ds.cells.extend(cells)
        return ds  
    
    def _to_dataset_helper_mesh(self, cells):
        for part in self._parts:
            cells.extend([Cell(i, part.ELEMENT_TYPE) for i in part._elements])
    
    def _to_dataset_helper_spring(self, cells):
        for s in self._springs:
            if s[2] is None:
                cells.append(Cell([s[1]//6], 1))
            else:
                cells.append(Cell([s[1]//6, s[2]//6], 3))

    def _to_dataset_helper_mass(self, cells):
        for m in self._masses:
            cells.append(Cell([s[1]]//6, 1))







#!/usr/bin/env python3

from .elements.shell import get_mass_matrix_shell, get_stiffness_matrix_shell
from .elements.beam import get_mass_matrix_beam, get_stiffness_matrix_beam
from .dataset import *


__all__ = ['PartBeam', 'PartShell']

class Part:
    """Base class for creating Part instances for finite element analysis.
    
    Parameters
    ----------
    nodes: 2-D np.ndarray
        The coordinates of the nodes. `nodes` should contain three columns
        representing x, y and z respectively.
    elements: 2-D np.ndarray
        The nodes of the element. Each row in `elements` contains the node
        indexes of the element. (Index starts from 0)
    section: Section object
        The section of the part.
    """
    
    ELEMENT_TYPE = 1
    
    def __init__(self, nodes, elements, section):
        self._nodes = nodes
        self._elements = elements
        self._section = section
        self._stiffness_matrix = None
        self._mass_matrix = None
        self._damping_matrix = None
        self._alpha, self._beta = 0,0
        self._is_initialized = False

    @property
    def nodes(self):
        return self._nodes
    @property
    def elements(self):
        return self._elements

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
        assert 0

    def _get_mass_matrix(self):
        assert 0

    def _get_damping_matrix(self):
        M = self._mass_matrix
        K = self._stiffness_matrix
        C = self._alpha * M + self._beta * K
        return C

    def set_damping(self, alpha, beta):
        """Set structural damping for the part."""
        self._uninitialize()
        self._alpha = alpha
        self._beta = beta
        return self

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

    def to_dataset(self):
        """Write part information to dataset.
        
        Returns
        -------
        ds: DataSet
            A DataSet object containing geometry information of the part.
        """
        points = [Point(i) for i in self._nodes]
        cells = [Cell(i, self.ELEMENT_TYPE) for i in self._elements]
        ds = DataSet(points, cells)
        return ds


class PartShell(Part):
    """Base class for creating shells for finite element analysis.
    
    Parameters
    ----------
    nodes: 2-D np.ndarray
        The coordinates of the nodes. `nodes` should contain three columns
        representing x, y and z respectively.
    elements: 2-D np.ndarray
        The nodes of the element. Each row in `elements` contains the node
        indexes of the element. (Index starts from 0)
    section: Section object
        The section of the shell.
    """
    
    ELEMENT_TYPE = 9
    
    def __init__(self, nodes, elements, section):
        super().__init__(nodes, elements, section)

    def _get_stiffness_matrix(self):
        return get_stiffness_matrix_shell(self._nodes, self._elements,
                                          self._section)

    def _get_mass_matrix(self):
        return get_mass_matrix_shell(self._nodes, self._elements,
                                     self._section)

class PartBeam(Part):
    """Base class for creating beams for finite element analysis.
    
    Parameters
    ----------
    nodes: 2-D np.ndarray
        The coordinates of the nodes. `nodes` should contain three columns
        representing x, y and z respectively.
    elements: 2-D np.ndarray
        The nodes of the element. Each row in `elements` contains the node
        indexes of the element. (Index starts from 0)
    section: Section object
        The section of the beam.
    """
    
    ELEMENT_TYPE = 3
    
    def __init__(self, nodes, elements, section, n1=None):
        super().__init__(nodes, elements, section)
        self._n1 = None
        self.set_direction(n1)

    def set_direction(self, n1=None):
        """Set the n1 direction of the cross section for the beam."""
        if n1 is None:
            self._n1 = (0,1,0)
        else:
            self._n1 = n1
        return self

    def _get_stiffness_matrix(self):
        return get_stiffness_matrix_beam(self._nodes, self._elements,
                                          self._section, self._n1)

    def _get_mass_matrix(self):
        return get_mass_matrix_beam(self._nodes, self._elements,
                                     self._section, self._n1)


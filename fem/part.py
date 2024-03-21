#!/usr/bin/env python3

from __future__ import annotations
import abc
from typing import TYPE_CHECKING, Optional, Type, Generic, TypeVar, \
    SupportsFloat


from .geometry import Vector, XYZType
from .elements import Beam2, MITC4, SolidX
from .dataset import Dataset, Mesh
from .section import Section, BeamSection, ShellSection, SolidSection

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix
    from .elements.base import ElementASM

__all__ = ['BeamPart', 'ShellPart', 'SolidPart']


class PartAssembleError(Exception):
    pass


SECTION = TypeVar('SECTION', bound=Section)


class Part(abc.ABC, Generic[SECTION]):
    """Abstract base class for part instances for finite element analysis.

    A part is something in a model shares the same section. It also
    contains the mesh information. One part instance can have only one
    assembler, which is used for assembling the mass and stiffness matrixes.
    Most assemblers can handle only one kind of cell, which means that
    the cells are restricted to having the same cell type. However, for
    assemblers which can handle multiple kinds of elements, `cells` can
    have different types (currently not implemented).

    Assemblers are derived from `ElementASM` in package `elements`.

    One-, two- and three- dimensional parts should derive this class
    and implement more detailed settings.

    Parameters
    ----------
    points : Points
        The coordinates of the points. The same `Points` object should be used
        for all `Part` objects in the model.
    cells : Cells
        Cell object containg the nodes of the cell. Note that different
        cell types may exist in the same part.
    section : Section object
        The section of the part.

    """

    @abc.abstractproperty
    def element_asm(self) -> Type[ElementASM]: ...

    def __init__(self, mesh: Mesh, section: SECTION):
        self._mesh = mesh
        self._section = section

        self._stiffness_matrix: csr_matrix | None = None
        self._mass_matrix: csr_matrix | None = None
        self._damping_matrix: csr_matrix | None = None

        self._alpha, self._beta = 0., 0.
        self._is_initialized = False

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def section(self) -> SECTION:
        return self._section

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
        return self.element_asm.get_stiffness_matrix(self.mesh,
                                                     self.section)

    def _get_mass_matrix(self):
        return self.element_asm.get_mass_matrix(self.mesh,
                                                self.section)

    def _get_damping_matrix(self) -> csr_matrix:
        if self._mass_matrix is None:
            raise RuntimeError(
                'mass matrix should be initialized before damping matrix')
        if self._stiffness_matrix is None:
            raise RuntimeError(
                'stiffness matrix should be initialized before damping matrix')
        M = self._mass_matrix
        K = self._stiffness_matrix
        C = self._alpha * M + self._beta * K
        return C

    def set_damping(self, alpha: SupportsFloat, beta: SupportsFloat) -> Part:
        """Set structural damping for the part."""
        self._uninitialize()
        self._alpha = float(alpha)
        self._beta = float(beta)
        return self

    @property
    def K(self) -> csr_matrix:
        self._initialize()
        return self._stiffness_matrix

    @property
    def M(self) -> csr_matrix:
        self._initialize()
        return self._mass_matrix

    @property
    def C(self) -> csr_matrix:
        self._initialize()
        return self._damping_matrix

    def to_dataset(self,
                   title: str = '',
                   time: Optional[SupportsFloat] = None) -> Dataset:
        """Write part information to dataset.

        Returns
        -------
        ds: Dataset
            A Dataset object containing geometry information of the part.
        title : string, optional
            The title of the dataset.
        time : float, optional
            Additional information for time step.
        """
        ds = Dataset(self.mesh, title, time)
        return ds


class BeamPart(Part[BeamSection]):
    """Class for creating beam parts for finite element analysis.

    Currently, only linear beam elements are supported.

    Parameters
    ----------
    points : Points
        The coordinates of the points. The same `Points` object should be used
        for all `Part` objects in the model.
    cells : Cells
        Cell object containg the nodes of the cell.
    section : Section object
        The section of the part.
    """

    element_asm = Beam2

    def __init__(self,
                 mesh: Mesh,
                 section: BeamSection,
                 n1: Vector | XYZType | None = None):
        super().__init__(mesh, section)
        self._n1: Vector
        self.set_direction(n1)

    def set_direction(self,
                      n1: Vector | XYZType | None = None):
        """Set the n1 direction of the cross section for the beam."""
        if n1 is None:
            _n1 = Vector(0., 1., 0.)
        elif isinstance(n1, tuple):
            _n1 = Vector(*(float(i) for i in n1))
        elif isinstance(n1, Vector):
            _n1 = n1
        else:
            raise TypeError('cannot convert n1 to Vector')
        self._n1 = _n1

    def _get_stiffness_matrix(self) -> csr_matrix:
        return self.element_asm.get_stiffness_matrix(self.mesh,
                                                     self.section,
                                                     self._n1)

    def _get_mass_matrix(self) -> csr_matrix:
        return self.element_asm.get_mass_matrix(self.mesh,
                                                self.section,
                                                self._n1)


class ShellPart(Part[ShellSection]):
    """Class for creating shell parts for finite element analysis.

    Currently, only MITC4 element is supported.

    Parameters
    ----------
    points : Points
        The coordinates of the points. The same `Points` object should be used
        for all `Part` objects in the model.
    cells : Cells
        Cell object containg the nodes of the cell.
    section : Section object
        The section of the part.
    """

    element_asm = MITC4


class SolidPart(Part[SolidSection]):
    """Class for creating solid parts for finite element analysis.

    Currently, only linear hexahedron element is supported.

    Parameters
    ----------
    points : Points
        The coordinates of the points. The same `Points` object should be used
        for all `Part` objects in the model.
    cells : Cells
        Cell object containg the nodes of the cell.
    section : Section object
        The section of the part.
    """
    element_asm = SolidX

#!/usr/bin/env python3

from __future__ import annotations
from typing import TYPE_CHECKING, Collection, Optional, Literal, \
    SupportsIndex

from .common import SequenceView, DOF
from .dataset import Points, Vertex, Line, Cell, Cells, Mesh


if TYPE_CHECKING:
    from .part import Part
    from scipy.sparse import csr_matrix


class Assembly:
    """Assemble the parts into an assembly.

    Each part of the assembly should have the same node list, whether they have
    been used in their element list.

    Parameters
    ----------
    parts : Collection of Part objects
        Parts in the assembly. The parts should have the same node list.
    """

    def __init__(self, parts: Collection[Part]):
        if len(parts) == 0:
            raise ValueError('assembly should contain at least one part')
        _parts = list(parts)
        points = _parts[0].mesh.points
        for p in _parts[1:]:
            if not points == p.mesh.points:
                raise ValueError(
                    'points array should be the same in each part')
        self._points = points
        self._parts = _parts
        self._stiffness_matrix: csr_matrix | None = None
        self._mass_matrix: csr_matrix | None = None
        self._damping_matrix: csr_matrix | None = None
        # The format of `self._springs` is:
        # list[tuple[stiffness, dof_index1, dof_index2]] if the spring
        # is connected between two nodes.
        # list[tuple[stiffness, dof_index1, None]] if spring is
        # connected to the ground.
        self._springs: list[tuple[float, int, int | None]] = []
        # The format of `self._masses` is:
        # list[tuple[mass, dof_index]]
        self._masses: list[tuple[float, int]] = []
        self._is_initialized = False

    @property
    def parts(self) -> list[Part]:
        return self._parts

    @property
    def points(self) -> Points:
        return self._points

    def add_spring(self,
                   k: float,
                   node1: int,
                   dof1: DOF,
                   node2: Optional[int] = None,
                   dof2: Optional[DOF] = None) -> Assembly:
        """Add a spring to the assembly.

        Users may define a spring connecting two nodes or connecting a node to
        the ground.

        Parameters
        ----------
        k : float
            Stiffness of the spring.
        node1 : int
            Index of the first node.
        dof1 : DOF
            The degree of freedom of the first node. Note that DOF is an
            IntEnum from 0 to 5.
        node2 : int, optional
            Index of the second node. If node2 is None, the spring is connected
            to the ground. (default to None)
        dof2 : DOF, optional
            The degree of freedom of the second node. If dof2 is None, dof2 is
            the same as dof1. Note that DOF is an IntEnum from 0 to 5. (default
            to None)
        """
        self._uninitialize()
        if node2 is None:
            self._springs.append((k, node1 * 6 + dof1, None))
        else:
            if dof2 is None:
                dof2 = dof1
            self._springs.append((k, node1 * 6 + dof1, node2 * 6 + dof2))
        return self

    @property
    def springs(self) -> SequenceView[tuple[float, int, int | None]]:
        return SequenceView(self._springs)

    def remove_spring(self, idx: Optional[SupportsIndex] = None) -> Assembly:
        """Remove a spring or springs from the assembly.

        Parameters
        ----------
        idx : int, optional
            Index of the spring to be removed. If `idx` is None, the all
            springs will be removed. (default to None)
        """
        self._uninitialize()
        if idx is None:
            self._springs.clear()
        else:
            self._springs.pop(idx)
        return self

    def add_mass(self, m: float, node: int, dof: DOF) -> Assembly:
        """Add a mass point to the assembly.

        Parameters
        ----------
        m : float
            The mass.
        node : int
            The index of the node.
        dof : DOF
            The degree of freedom of the node.  Note that DOF is an
            IntEnum from 0 to 5.
        """
        self._uninitialize()
        self._masses.append((m, node*6+dof))
        return self

    @property
    def masses(self) -> SequenceView[tuple[float, int]]:
        return SequenceView(self._masses)

    def remove_mass(self, idx: Optional[SupportsIndex] = None) -> Assembly:
        """Remove mass point from the assembly.

        Parameters
        ----------
        idx : int, optional
            Index of the mass to be removed. If `idx` is None, all the mass
            points will be removed. (default to None)
        """
        self._uninitialize()
        if idx is None:
            self._masses.clear()
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

    def _get_stiffness_matrix(self) -> csr_matrix:
        K = self._parts[0].K.tocsr().copy()
        for p in self._parts[1:]:
            K += p.K.tocsr()
        # Change the type of K to lil matrix to avoid the following
        # warning: `SparseEfficiencyWarning: Changing the sparsity
        # structure of a csr_matrix is expensive. lil_matrix is more
        # efficient.`
        K = K.tolil()
        for k, dof1, dof2 in self._springs:
            if dof2 is None:
                K[dof1, dof1] += k
            else:
                K[dof1, dof1] += k
                K[dof1, dof2] += -k
                K[dof2, dof1] += -k
                K[dof2, dof2] += k
        return K.tocsr()

    def _get_mass_matrix(self) -> csr_matrix:
        M = self._parts[0].M.tocsr().copy()
        for p in self._parts[1:]:
            M += p.M.tocsr()
        for m, dof in self._masses:
            M[dof, dof] += m
        return M

    def _get_damping_matrix(self) -> csr_matrix:
        C = self._parts[0].C.tocsr().copy()
        for p in self._parts[1:]:
            C += p.C.tocsr()
        return C

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

    def to_mesh(self,
                which: Literal['all', 'mesh', 'spring', 'mass'] = 'all'
                ) -> Mesh:
        """Write assembly information to dataset.

        Parameters
        ----------
        which: 'all' | 'mesh' | 'spring' | 'mass'
            Define which part of the assembly to export.

        Returns
        -------
        ds: Dataset
            A Dataset object containing geometry information of the assembly.
        """
        _which = which.lower()
        points = self._points
        cell_list: list[Cell] = []
        if _which == 'all':
            self._to_dataset_helper_mesh(cell_list)
            self._to_dataset_helper_spring(cell_list)
            self._to_dataset_helper_mass(cell_list)
        elif _which == 'mesh':
            self._to_dataset_helper_mesh(cell_list)
        elif _which == 'spring':
            self._to_dataset_helper_spring(cell_list)
        elif _which == 'mass':
            self._to_dataset_helper_mass(cell_list)
        else:
            raise AttributeError("'which' should be in 'all' | 'mesh' | 'sprin"
                                 "g' | 'mass'")
        cells = Cells(cell_list)
        mesh = Mesh(points, cells)
        return mesh

    def _to_dataset_helper_mesh(self, cell_list: list[Cell]):
        for part in self._parts:
            cell_list.extend(part.mesh.cells)

    def _to_dataset_helper_spring(self, cell_list: list[Cell]):
        for s in self._springs:
            if s[2] is None:
                cell_list.append(Vertex(s[1]//6))
            else:
                cell_list.append(Line(s[1]//6, s[2]//6))

    def _to_dataset_helper_mass(self, cell_list: list[Cell]):
        for m in self._masses:
            cell_list.append(Vertex(m[1]//6))

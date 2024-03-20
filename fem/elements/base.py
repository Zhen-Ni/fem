#!/usr/bin/env python3

from __future__ import annotations
import abc
from typing import TYPE_CHECKING, TypeVar, Generic
from scipy.sparse import csr_matrix, coo_matrix

from ..section import Section, BeamSection, ShellSection, SolidSection

if TYPE_CHECKING:
    import numpy.typing as npt
    from ..mesh import Points, Cells
    from ..common import Vector


__all__ = ('Beam2', 'MITC4', 'SolidX')


SECTION = TypeVar('SECTION', bound=Section)


class ElementASM(abc.ABC, Generic[SECTION]):
    """Abstract base class for assemble element matrixes."""

    @abc.abstractstaticmethod
    def get_mass_matrix(points: Points,
                        cells: Cells,
                        section: SECTION,
                        **kwargs) -> csr_matrix: ...

    @abc.abstractstaticmethod
    def get_stiffness_matrix(points: Points,
                             cells: Cells,
                             section: SECTION,
                             **kwargs) -> csr_matrix: ...


class Beam2(ElementASM[BeamSection]):

    @staticmethod
    def get_mass_matrix(points: Points,
                        cells: Cells,
                        section: BeamSection,
                        n1: Vector) -> csr_matrix:
        from .beam import get_mass_matrix_beam
        nodes = points.to_array()
        elements = cells.to_array()
        return get_mass_matrix_beam(nodes, elements, section,
                                    [n1.x, n1.y, n1.z]).tocsr()

    @staticmethod
    def get_stiffness_matrix(points: Points,
                             cells: Cells,
                             section: BeamSection,
                             n1: Vector) -> csr_matrix:
        from .beam import get_stiffness_matrix_beam
        nodes = points.to_array()
        elements = cells.to_array()
        return get_stiffness_matrix_beam(nodes, elements, section,
                                         [n1.x, n1.y, n1.z]).tocsr()


class MITC4(ElementASM[ShellSection]):

    @staticmethod
    def get_mass_matrix(points: Points,
                        cells: Cells,
                        section: ShellSection,
                        ) -> csr_matrix:
        from .mitc import get_mass_matrix
        nodes = points.to_array()
        elements = cells.to_array()
        return get_mass_matrix(section.material.rho,
                               section.h,
                               elements,
                               nodes).tocsr()

    @staticmethod
    def get_stiffness_matrix(points: Points,
                             cells: Cells,
                             section: ShellSection
                             ) -> csr_matrix:
        from .mitc import get_stiffness_matrix
        nodes = points.to_array()
        elements = cells.to_array()
        return get_stiffness_matrix(section.material.E,
                                    section.material.rho,
                                    section.h,
                                    elements,
                                    nodes).tocsr()


class SolidX(ElementASM[SolidSection]):

    @staticmethod
    def get_mass_matrix(points: Points,
                        cells: Cells,
                        section: SolidSection,
                        ) -> csr_matrix:
        from .solidx import get_mass_matrix_hexahedron
        mat = get_mass_matrix_hexahedron(points, cells, section)
        mat = expand_dof(mat)
        return mat.tocsr()

    @staticmethod
    def get_stiffness_matrix(points: Points,
                             cells: Cells,
                             section: SolidSection
                             ) -> csr_matrix:
        from .solidx import get_stiffness_matrix_hexahedron
        mat = get_stiffness_matrix_hexahedron(points, cells, section)
        mat = expand_dof(mat)
        return mat.tocsr()


def expand_dof(mat: coo_matrix) -> coo_matrix:
    """Add rotational dofs into assembled matrix."""
    shape = mat.shape
    row = _expand_dof_helper(mat.row)
    col = _expand_dof_helper(mat.col)
    value = mat.data
    res = coo_matrix((value, (row, col)), [shape[0] * 2, shape[1] * 2])
    return res


def _expand_dof_helper(arr: npt.NDArray) -> npt.NDArray:
    node_id = arr // 3
    node_dof = arr % 3
    return node_id * 6 + node_dof

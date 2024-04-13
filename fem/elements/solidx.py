#!/usr/bin/env python3

from __future__ import annotations

import abc
import numpy as np
import skfem
from skfem.models.elasticity import linear_elasticity, lame_parameters
from skfem.helpers import dot
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from scipy.sparse import coo_matrix
    from ..dataset import Mesh
    from ..section import SolidSection


class MeshMap(abc.ABC):
    @property
    @abc.abstractmethod
    def index_map(self) -> tuple[int, ...]: ...

    @property
    @abc.abstractmethod
    def elem(self) -> Type[skfem.Element]: ...

    @classmethod
    def to_skfem(cls, mesh: Mesh) -> skfem.Mesh:
        t_list = []
        for cell in mesh.cells:
            elem = []
            for j in cls.index_map:
                elem.append(cell.nodes[j])
            t_list.append(elem)
        doflocs = mesh.points.to_array().T
        t = np.array(t_list).T
        # Use C_CONTIGUOUS array is more performant in dimension-based
        # slices, as indicated by skfem.mesh.
        doflocs = np.ascontiguousarray(doflocs)
        t = np.ascontiguousarray(t)
        mesh = skfem.Mesh3D(doflocs, t)
        mesh.elem = cls.elem
        return mesh


class HexahedronMap(MeshMap):
    index_map = (3, 0, 7, 2, 4, 1, 6, 5)
    elem = skfem.MeshHex1.elem


@skfem.BilinearForm
def mass_form(u, v, w):
    return dot(w.rho * u, v)


def get_mass_matrix_hexahedron(mesh: Mesh,
                               section: SolidSection
                               ) -> coo_matrix:
    mesh = HexahedronMap.to_skfem(mesh)
    e1 = skfem.ElementHex1()
    e = skfem.ElementVector(e1)
    basis = skfem.Basis(mesh, e)
    rho = section.material.rho
    M = skfem.asm(mass_form, basis, rho=rho)
    return M.tocoo()


def get_stiffness_matrix_hexahedron(mesh: Mesh,
                                    section: SolidSection
                                    ) -> coo_matrix:
    mesh = HexahedronMap.to_skfem(mesh)
    e1 = skfem.ElementHex1()
    e = skfem.ElementVector(e1)
    basis = skfem.Basis(mesh, e)
    E = section.material.E
    nu = section.material.nu
    K = skfem.asm(linear_elasticity(*lame_parameters(E, nu)), basis)
    return K.tocoo()

#!/usr/bin/env python3


from __future__ import annotations
from typing import TYPE_CHECKING, Optional, SupportsIndex

import numpy as np

from .common import empty, SequenceView
from .geometry import Vector, XYZType
from .dataset import Points, Dataset, FloatArrayField, Mesh

if TYPE_CHECKING:
    import numpy.typing as npt
    from scipy.sparse import csr_matrix
    from .assembly import Assembly


class Model:
    """Create a model for finite element analysis.

    Forces may be added to Model objects.

    Parameters
    ----------
    assembly : Assembly object
        The assembly to be analyzed.
    """

    def __init__(self, assembly: Assembly):
        self._assembly = assembly
        # The format of `self._forces` is: list[tuple[force, dof_index]]
        self._forces: list[tuple[float, int]] = []
        # The format of `self._forces_gravity` is:
        # list[tuple[gravity_acceration, normalized_direction]]
        self._forces_gravity: list[tuple[float, Vector]] = []
        self._force_vector: npt.NDArray[np.float64] | None = None
        self._is_initialized = False

    @property
    def points(self) -> Points:
        return self.assembly.points

    @property
    def assembly(self) -> Assembly:
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

    def add_force(self, f: float, node: int, dof: int) -> Model:
        """Add force to the model.

        Parameters
        ----------
        f : float
            Force.
        node : int
            The index of the node.
        dof : int
            The degree of freedom of the node.
        """
        self._uninitialize()
        self._forces.append((f, node * 6 + dof))
        return self

    def add_gravity(self,
                    g: float,
                    direction: (Vector | XYZType) = Vector(z=-1)
                    ) -> Model:
        """Add gravity to the model.

        Parameters
        ----------
        g : float
            Gravity.
        direction : Vector or tuple of three floats
            The direction of the gravity.
        """
        self._uninitialize()
        if not isinstance(direction, Vector):
            direction = Vector(*(float(i) for i in direction))
        self._forces_gravity.append((g, direction.normalize()))
        return self

    @property
    def forces(self) -> SequenceView[tuple[float, int]]:
        return SequenceView(self._forces)

    @property
    def gravities(self) -> SequenceView[tuple[float, Vector]]:
        return SequenceView(self._forces_gravity)

    def remove_force(self, idx: Optional[SupportsIndex] = None) -> Model:
        """Remove force from the assembly.

        Parameters
        ----------
        idx : int, optional
            Index of the force to be removed. If `idx` is None, all the forces
            will be removed. (default to None)
        """
        self._uninitialize()
        if idx is None:
            self._forces.clear()
        else:
            self._forces.pop(idx)
        return self

    def remove_gravity(self, idx: Optional[SupportsIndex] = None) -> Model:
        """Remove gravity from the assembly.

        Parameters
        ----------
        idx: int, optional
            Index of the gravity to be removed. If `idx` is None, all the
            gravities will be removed. (default to None)
        """
        self._uninitialize()
        if idx is None:
            self._forces_gravity.clear()
        else:
            self._forces_gravity.pop(idx)
        return self

    def _get_force_vector(self) -> npt.NDArray[np.float64]:
        F = np.zeros(self._assembly.K.shape[0], dtype=float)
        for f, dof in self._forces:
            F[dof] += f
        mass_component = self._assembly.M.diagonal()
        for g, direction in self._forces_gravity:
            gx, gy, gz = g * direction
            for i in range(len(F) // 6):
                F[i * 6] += gx * mass_component[i * 6]
                F[i * 6 + 1] += gy * mass_component[i * 6 + 1]
                F[i * 6 + 2] += gz * mass_component[i * 6 + 2]
        return F

    @property
    def M(self) -> csr_matrix:
        return self._assembly.M

    @property
    def K(self) -> csr_matrix:
        return self._assembly.K

    @property
    def C(self) -> csr_matrix:
        return self._assembly.C

    @property
    def F(self) -> npt.NDArray[np.float64]:
        self._initialize()
        # self._force_vector is set in self._initialize, thus it
        # should be an array.
        assert self._force_vector is not None
        return self._force_vector

    def to_mesh(self) -> Mesh:
        return self.assembly.to_mesh()

    @property
    def to_dataset(self) -> Dataset:
        mesh = self.assembly.to_mesh()
        dataset = Dataset(mesh)

        force = empty(len(self.points), 3)
        moment = empty(len(self.points), 3)
        for (f, dof) in self._forces:
            index = dof // 6
            direction = index % 6
            if direction < 3:
                force[index][direction] += f
            else:
                moment[index][direction - 3] += f
        force_field = FloatArrayField(force)
        moment_field = FloatArrayField(moment)
        dataset.point_data['force'] = force_field
        dataset.point_data['moment'] = moment_field
        gravity = [0., 0., 0.]
        for g, v in self._forces_gravity:
            gravity[0] += g * v.x
            gravity[1] += g * v.y
            gravity[2] += g * v.z
        gravity_field = FloatArrayField([gravity] * len(dataset.cells))
        dataset.cell_data['gravity'] = gravity_field
        return dataset

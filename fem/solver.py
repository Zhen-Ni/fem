#!/usr/bin/env python3

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, final, Sequence, SupportsIndex, overload,\
    Iterator, Optional, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix, csr_matrix, spmatrix
from scipy.sparse.linalg import eigsh, spsolve

from .common import SequenceView, Readonly, InhertSlotsABCMeta
from .geometry import Vector
from .dataset import Points, Mesh, Dataset, Field, ScalarField, ArrayField


if TYPE_CHECKING:
    from .model import Model


__all__ = 'StaticSolver', 'ModalSolver'


class SolverError(Exception):
    pass


class Frame(Readonly, metaclass=InhertSlotsABCMeta):
    """A frame of the solution.

    A frame stores information of solving one FEM equation, can be a
    modal, a timestep of a transient analysis. It gets the model mesh
    (before deformation) and the displacement vector (nodal solution)
    as fundamental input. The nodal solution includes both
    translational and rotaional displacement, thus has 6-th the size
    of the number of nodes defined in the mesh. Nodal solution will be
    converted into an `ArrayField` object while initializing. Label
    and label values are used to define the type and the status of the
    frame. For example, in modal analysis, label can be set to
    'frequency' and its values is the corresponding frequency.

    Other field variables and scalar variables are optional, and can
    be given by `scalars` and `fields`. Note that unlike
    nodal_solution, which is converted into `Field` object
    automatically, when defining other field variables, users should
    convert it into `Field` objects first.

    Parameters
    ----------
    mesh : Mesh
        The undeformed mesh of the finite element model.
    nodal_solution : npt.NDArray
        Should be 2-dimensional array with shape (6 *
        len(mesh.points), ) to represent 6 DOFs of each node. The
        order of the dofs of the nodes follows that given by fem.DOF.
    label : str, optional
        The label of the frame. Can be set to an empty string. defaults
        to ''.
    label_value : float, optional
        The corresponding value for label. defaults to nan.
    scalars : dict[str, float], optional
        Other optional scalar values to be stored in the frame.
    fields : dict[str, Field], optional
        Other optional scalar of vector fields to be stored in the frame.

    """

    __slots__ = ('_mesh', '_nodal_solution', '_label', '_label_value',
                 '_scalars', '_fields')

    def __init__(self,
                 mesh: Mesh,
                 nodal_solution: npt.NDArray,
                 label: str = '',
                 label_value: float = float('nan'),
                 *,
                 scalars: dict[str, float] | None = None,
                 fields: Optional[dict[str, Field]] = None):
        self._mesh = mesh
        self._nodal_solution = ArrayField.from_array(
            nodal_solution.reshape(-1, 6))
        if len(self._mesh.points) != len(self._nodal_solution):
            raise ValueError('shape of nodal_solution not correct')
        self._label = label
        self._label_value = label_value
        self._scalars = {} if scalars is None else scalars
        self._fields = {} if fields is None else fields
        for key in self._scalars.keys():
            if key in self._fields:
                raise KeyError('scalars and fields should have unique keys')
        for key in self._scalars.keys() | self._fields.keys():
            if key == label:
                raise KeyError('keys of `scalars` and `fields`'
                               ' can not be the same with `label`')
            if key in ('translation', 'rotation'):
                raise KeyError('keys of `scalars` and `fields`'
                               f' can not be "{key}"')

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def label(self) -> str:
        return self._label

    @property
    def label_value(self) -> float:
        return self._label_value

    @property
    def nodal_solution(self) -> ArrayField:
        return self._nodal_solution

    def __getitem__(self, key: str) -> float | Field:
        if key == self.label:
            return self.label_value
        if key == 'translation':
            return self.translation()
        if key == 'rotation':
            return self.rotation()
        if self._fields.get(key, None) is not None:
            return self._fields[key]
        if self._scalars.get(key, None) is not None:
            return self._scalars[key]
        raise KeyError(f'{self} has no scalars or fields with key `{key}`')

    @overload
    def dof(self, index: SupportsIndex) -> ScalarField: ...

    @overload
    def dof(self, index: Sequence[SupportsIndex]) -> ArrayField: ...

    def dof(self, index):
        """Get the corresponding dof value of nodal solution."""
        return self.nodal_solution.dof(index)

    def translation(self) -> ArrayField:
        """Get the translational dofs of nodal solution."""
        return self.dof([0, 1, 2])

    def rotation(self) -> ArrayField:
        """Get the rotational dofs of nodal solution."""
        return self.dof([3, 4, 5])

    def deformed_mesh(self) -> Mesh:
        """Get deformed mesh."""
        points = [p + Vector(*v) for p, v in
                  zip(self.mesh.points, self.translation())]
        return Mesh(Points(points), self.mesh.cells)

    def to_dataset(self) -> Dataset:
        """Convert frame to dataset objects.

        The scalars are ignores as they are not supported in dataset
        objects.
        """
        ds = Dataset(self.mesh, title=self.label, time=self.label_value)
        ds.point_data['translation'] = self.translation()
        ds.point_data['rotation'] = self.rotation()
        ds.point_data.update(self._fields)
        return ds


@final
class Step:
    """A collection of frames storing the analysis result.
    """

    __slots__ = ('__frames')

    def __init__(self, frames: Sequence[Frame] | None = None):
        self.__frames = [] if frames is None else list(frames)

    def append(self, frame: Frame):
        self.__frames.append(frame)

    def __repr__(self):
        return "Step object with {n} frames".format(n=len(self.__frames))

    def __len__(self) -> int:
        return len(self.__frames)

    def __iter__(self) -> Iterator[Frame]:
        for f in self.__frames:
            yield f

    @overload
    def __getitem__(self, index: SupportsIndex) -> Frame: ...

    @overload
    def __getitem__(self, index: slice | Sequence[SupportsIndex]
                    ) -> SequenceView[Frame]: ...

    def __getitem__(self, index):
        return SequenceView(self.__frames)[index]

    def to_datasets(self) -> list[Dataset]:
        """Convert each frame to datasets."""
        dss = [f.to_dataset() for f in self]
        return dss


class Solver(abc.ABC):

    slots = ('__model')

    def __init__(self, model: Model):
        self.__model = model

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} object>"

    @property
    def model(self) -> Model:
        return self.__model

    @abc.abstractmethod
    def solve(self) -> Step: ...


class ModalSolver(Solver):
    """Get the modal of the model.

    This solver treats fully real symmetric matrixes and no damping is
    considered.
    """

    def __init__(self,
                 model: Model,
                 order: int = 20,
                 frequency_shift: float = 0.0):
        super().__init__(model)
        self._order = order
        self._frequency_shift = frequency_shift

    def set_order(self, order: int):
        self._order = order

    def set_frency_shift(self, frequency_shift: float):
        self._frequency_shift = frequency_shift

    def solve(self) -> Step:
        K = self.model.K
        M = self.model.M
        # Needs to drop inactive dofs to avoid singularity
        (K, M), active_dof = drop_inactive_dof(K, M)
        freq, mode_shape = eigsh(K, self._order, M, self._frequency_shift,)
        freq = np.sqrt(freq.real) / 2 / np.pi
        index_array = np.argsort(freq.real)
        step = Step()
        for idx in index_array:
            f = freq[idx]
            x = mode_shape.T[idx]
            x = recover_inactive_dof(x, active_dof)
            step.append(Frame(self.model.to_mesh(), x, 'frequency', f))
        return step


class StaticSolver(Solver):
    """Get the static deformation of the model."""

    def __init__(self, model: Model):
        super().__init__(model)

    def solve(self) -> Step:
        (K,), active_dof = drop_inactive_dof(self.model.K)
        if self.model.F[~active_dof].any():
            raise SolverError('StaticSolver error: force applied '
                              'on inactive dof')
        displacement = spsolve(K, self.model.F[active_dof])
        return Step([Frame(self.model.to_mesh(),
                           recover_inactive_dof(displacement,
                                                active_dof),
                           'static', 1.0)])


def recover_inactive_dof(x: npt.NDArray,
                         active_dof: npt.NDArray[bool]
                         ) -> npt.NDArray:
    """Recover inactive dofs of nodal solution `x`.

    The input `active_dof` is that return by `drop_inactive_dof`,
    which indicates the inactive dofs in the original finite element
    system.
    """
    x_recovered = np.zeros(len(active_dof), dtype=x.dtype)
    x_recovered[active_dof] = x
    return x_recovered


def drop_inactive_dof(*mats: spmatrix
                      ) -> tuple[list[csr_matrix],
                                 npt.NDArray[bool]]:
    """Drop inactive dofs in finite element matrixes `mats`.

    A list of sparse matrixes with the same shape is given. The sparse
    matrixes should have same number of rows and columns. Typical
    example of input matrixes are [M, K] or [M, C, K] where M, C, K
    are mass matrix, damping matrix and stiffness matrix,
    respectively. The inactive dof `i` is defined that for matrix A,
    A[i, :] and A[:, i] are all zeros (explicit zeros are ignored). If
    a dof `i` is inactive in all input matrixes, it is removed by this
    function. For the original input matrix, the dofs to be removed
    are marked in another array `active_dof`, which is a ndarray of
    booleans. The length of the array is the same as the size of the
    original input matrixes, thus each of its element indicates a
    dof. Thus, all remaining dofs are marked as `True` and removed
    dofs are marked as `False` in the corresponding position of
    `active_dof`.

    Parameters
    ----------
    mats : list of sparse matrixes
        A list of sparse matrixes of shape n * n.

    Outputs
    -------
    results : list of csr_matrix
        The corresponding sparse matrixes where empty dofs are moved.
    active_dof : ndarray of bool
        An array of shape n, True value indicates the corresponding
        dof is kept, and False value indicates the dof is removed.
    """
    if len(mats) == 0:
        raise ValueError('at least one sparse matrix should be provided')
    active_dof = _get_active_dof(mats[0])
    for mat in mats[1:]:
        active_dof |= _get_active_dof(mat)  # type: ignore
    results = [_trim_dof(m, active_dof) for m in mats]
    return results, active_dof


def _get_active_dof(mat: spmatrix) -> npt.NDArray[bool]:
    """For given sparse matrix with shape n x n, get an array
    indicating where corresponding row and columns are not all zeros.
    """
    nonzero_rows = _get_nonzero_inner(mat.tocsr())
    nonzero_cols = _get_nonzero_inner(mat.tocsc())
    return nonzero_cols | nonzero_rows  # type: ignore


def _trim_dof(mat: spmatrix, keep: npt.NDArray[bool]) -> csr_matrix:
    """Drop rows and columns indicated by `False` in `keep`."""
    mat = _trim_inner(mat.tocsc(), keep)
    mat = _trim_inner(mat.tocsr(), keep)
    return mat


_Spmat = TypeVar('_Spmat', csc_matrix, csr_matrix)


def _trim_inner(mat: _Spmat, keep: npt.NDArray[bool]) -> _Spmat:
    """Drop rows (csr_matrix) or columns (csc_matrix) or of the matrix
    indicated by `False` in `keep`, and keep rows or columns indicated
    by `True` in `keep`.

    This is hinted by [[https://github.com/scipy/scipy/issues/6754]].
    """
    new_indptr = mat.indptr[np.append(True, keep)]
    if isinstance(mat, csr_matrix):
        new_shape = (np.count_nonzero(keep), mat.shape[1])
    elif isinstance(mat, csc_matrix):
        new_shape = (mat.shape[0], np.count_nonzero(keep))
    new_mat = type(mat)((mat.data, mat.indices, new_indptr),
                        shape=new_shape)
    return new_mat


def _get_nonzero_inner(mat: csc_matrix | csr_matrix) -> npt.NDArray[bool]:
    """Get non-empty row indexes for csr matrix, and column indexes
    for csc matrix.

    This is hinted by [[https://github.com/scipy/scipy/issues/6754]].
    """
    inner_nnz = np.diff(mat.indptr)
    nonzero = (inner_nnz != 0)
    return nonzero

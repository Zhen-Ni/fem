#!/usr/bin/env python3

from __future__ import annotations
import abc
from typing import overload, Type, Generic, TypeVar, Optional, \
    SupportsIndex, SupportsFloat, SupportsInt
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt

from .common import CellType, SequenceView, Readonly, InhertSlotsABCMeta, \
    XYZBase

try:
    # Available for sys.version > "3.11".
    from typing import Self
except ImportError:
    from typing_extensions import Self


__all__ = ('Point', 'Points',
           'CellType', 'Vertex', 'Line', 'Quad', 'Hexahedron', 'Cells',
           'ScalarField', 'VectorField',
           'Dataset')


T = TypeVar('T')


class DatasetBase(Sequence,
                  Readonly,
                  Generic[T],
                  metaclass=InhertSlotsABCMeta):
    """Base class for Points, Nodes, Fields, etc...

    This is a wrapper for `SequenceView` type, which is immutable and
    can be used to store field information in fem.
    """

    __slots__ = '__storage'

    @abc.abstractmethod
    def __init__(self, storage: list[T]):
        """We'd like to take full control of `storage`, so make sure
        `storage` is copied in the derived class, so that it has no
        reference outside the class.
        """
        self.__storage = SequenceView(storage)

    @abc.abstractmethod
    def __repr__(self) -> str:
        return f"{self.__class__.__name__} object with {len(self)} data"

    def __str__(self) -> str:
        header = repr(self) + ':\n'
        body = '\n'.join([f'{i}\t{line}' for i, line in enumerate(self)])
        ending = ''
        return header + body + ending

    @overload
    def __getitem__(self, index: SupportsIndex) -> T: ...

    @overload
    def __getitem__(self: Self,
                    index: slice | Sequence[SupportsIndex]) -> Self: ...

    def __getitem__(self, index):
        if isinstance(index, SupportsIndex):
            return self.__storage[index]
        elif isinstance(index, (slice, Sequence)):
            return self.__class__(self.__storage[index])
        raise TypeError(
            f'{self.__class__.__name__} indices must be integers,'
            f' slices, or a sequence of int, not {type(index)}')

    def __len__(self) -> int:
        return len(self.__storage)

    def __eq__(self, other) -> bool:
        if not type(self) == type(other):
            return False
        if len(self) != len(other):
            return False
        for p1, p2 in zip(self, other):
            if p1 != p2:
                return False
        return True

    def to_list(self) -> list[T]:
        return list(self)


class Point(XYZBase):
    def __repr__(self):
        return f"Point: ({self.x}, {self.y}, {self.z})"

    def coordinates(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


class Points(DatasetBase[Point]):
    """Class for storing the nodes of a mesh."""

    def __init__(self, point_list: Sequence[Point]):
        # Make sure point_list is only itered once in case it is an iterator.
        storage_list = []
        for p in point_list:
            if isinstance(p, Point):
                storage_list.append(p)
            else:
                raise TypeError('elements in point_list should be `Point`,'
                                f' got {type(p)}')
        super().__init__(storage_list)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} object with {len(self)} points"

    @staticmethod
    def from_array(array: npt.ArrayLike) -> Points:
        """Generate a Points object from the given array.

        The array should be 2-dimensional and the number of columns
        should be 3, which represents the x-, y- and z- coordinates,
        respectively.

        Parameters
        ----------
        array : npt.ArrayLike
            The array containing the coordinates.
        """
        return Points([Point(*row) for row in np.asarray(array)])

    def to_array(self) -> npt.NDArray[float]:
        return np.array([[p.x, p.y, p.z] for p in self])

    @property
    def x(self) -> list[float]:
        return [p.x for p in self]

    @property
    def y(self) -> list[float]:
        return [p.y for p in self]

    @property
    def z(self) -> list[float]:
        return [p.z for p in self]


class Cell(Readonly, abc.ABC):
    """A single cell of specific type.

    The cell types are the same with that of VTK types.
    """

    __slots__ = '__nodes',

    def __init__(self, *nodes: SupportsInt):
        if len(nodes) != self.size:
            raise TypeError(f'expect {self.size} nodes, got {len(nodes)}')
        self.__nodes = tuple((int(i) for i in nodes))

    @abc.abstractproperty
    def id(self) -> CellType: ...

    @abc.abstractproperty
    def size(self) -> int: ...

    @property
    def nodes(self) -> tuple[int, ...]:
        return self.__nodes

    def __repr__(self) -> str:
        return f"Cell: {self.__class__.__name__}{self.nodes}"


class Vertex(Cell):
    id = CellType.VERTEX
    size = 1


class Line(Cell):
    """Two-node linear line cell.

    The nodes are as follows:
    0--1
    """
    id = CellType.LINE
    size = 2


class Quad(Cell):
    """Four-node linear facet cell.

    The nodes are as follows:
    3--2
    |  |
    0--1
    """
    id = CellType.QUAD
    size = 4


class Tetra(Cell):
    r"""Four-node linear volume cell.

    The nodes are as follows:
           3
         / | \
       /   |   \
     /     |     \
    2------+------1
     \     |     /
       \   |   /
         \ | /
           0
    """
    id = CellType.TETRA
    size = 4


class Hexahedron(Cell):
    """Eight-node linear volume cell.

    The nodes are as follows:
       7-----6
      /|    /|
     / |   / |
    4--+--5  |
    |  3--+--2
    | /   | /
    |/    |/
    0-----1
    """
    id = CellType.HEXAHEDRON
    size = 8


class Cells(DatasetBase[Cell]):
    """Class for stroing cells info of a mesh.

    Each cell is formed by connection of its nodes. The sequence
    of the nodes are defined the same as that of VTK cells. The nodes
    are stored by their node ids.
    """

    def __init__(self, cell_list: Sequence[Cell]):
        # Make sure point_list is only itered once in case it is an iterator.
        storage_list = []
        for c in cell_list:
            if isinstance(c, Cell):
                storage_list.append(c)
            else:
                raise TypeError('elements in point_list should be `Cell`,'
                                f' got {type(c)}')
        super().__init__(storage_list)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} object with {len(self)} cells"

    @staticmethod
    def from_array(cell_type: Type[Cell],
                   nodes_array: npt.ArrayLike) -> Cells:
        """Generate an Cells object from an array.

        This method is used to get `Cell`s object if the cells are of
        the same type. Each row in node_array represents a cell. The
        number of columns of nodes_array should be the same with
        cell_type.size.

        Parameters
        ----------
        cell_type : Subclass of Cell
            Define cell type.
        nodes_array : npt.ArrayLike
            Array defining the nodes of the cells.
        """
        return Cells([cell_type(*i) for i in np.asarray(nodes_array)])

    def same_cell_type(self) -> Type[Cell] | None:
        """Get cell type if all cells are of the same type.

        If all cells are the same type, return a single cell
        type. Return None if cells are of different types or
        the `Cells` object is empty.
        """
        if len(self) == 0:
            return None
        t = type(self[0])
        for c in self:
            if t != type(c):
                return None
        return t

    def to_array(self) -> npt.NDArray:
        res = np.array([list(cell.nodes) for cell in self])
        if res.dtype == object:
            raise TypeError('all cells should have same number of nodes')
        return res


class Field(DatasetBase[T]):
    """Base class for field data in the Dataset.

    An instance of the `Field` class defines a named field for point
    data or cell data. The field data is a sequence of scalars or
    vectors. Users should use `ScalarField` or `VectorField` for
    instantiation `Field` objects.

    Parameters
    ----------
    data_name : string
        Name of the field.
    data : iterable
        Data for each point or cell.

    See Alse
    --------
    ScalarField, VectorField
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} object with {len(self)} data"


class ScalarField(Field[float]):
    """Field for scalar data.

    Parameters
    ----------
    scalar : iterable
        Data for each point or cell.

    See Alse
    --------
    VectorField
    """

    def __init__(self, scalars: Sequence[SupportsFloat]):
        # Make sure point_list is only itered once in case it is an iterator.
        storage_list = []
        for p in scalars:
            try:
                storage_list.append(float(p))
            except (TypeError, ValueError):
                raise TypeError('elements in scalars should be float,'
                                f' got {type(p)}')
        super().__init__(storage_list)


class VectorField(Field[tuple[float, ...]]):
    """ Field for vector data.

    Parameters
    ----------
    data : iterable
        Data for each point or cell.
    dimension : int, optional
        Length of the field data for each point or cell. (default to 3)

    See Alse
    --------
    ScalarField
    """

    __slots__ = ('_dimension',)

    def __init__(self,
                 vectors: Sequence[Sequence[SupportsFloat]],
                 dimension: SupportsInt = 3):
        _dimension = int(dimension)
        # Make sure point_list is only itered once in case it is an iterator.
        storage_list = []
        for v in vectors:
            try:
                elem = [float(i) for i in v]
                if not len(elem) == _dimension:
                    raise ValueError('shape of `vectors` not correct')
                storage_list.append(tuple(elem))
            except Exception:
                raise TypeError('elements in vectors should be'
                                ' a sequence of float')
        super().__init__(storage_list)
        self._dimension = _dimension

    @property
    def dimension(self) -> int:
        return self._dimension


class Dataset:
    """Class for mesh geometry and the corresponding field data.

    Dataset class corresponds to VTK file formats using unstructured mesh. It
    stores the points and cells to construct the geometry. Field data can be
    added to the instances to represent the point data or cell data.

    Parameters
    ----------
    points : Points
        The nodes of a mesh.
    cells : Cells, optional
        The connectivity of a mesh.
    title : string, optional
        The title of the dataset.
    time : float, optional
        Additional information for time step.
    """

    def __init__(self,
                 points: Points,
                 cells: Optional[Cells] = None,
                 title: str = '',
                 time: Optional[SupportsFloat] = None):
        self._points = points
        self._cells = Cells([]) if cells is None else cells

        # self._title and self._time are set by attrbute setters.
        self.title = title
        self.time = time

        self._point_data: dict[str, Field] = {}
        self._cell_data: dict[str, Field] = {}

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, title: str):
        """Set the title for the dataset."""
        self._title = str(title)

    @property
    def time(self) -> Optional[float]:
        return self._time

    @time.setter
    def time(self, time: Optional[SupportsFloat]):
        self._time = None if time is None else float(time)

    @property
    def points(self) -> Points:
        """List of points."""
        return self._points

    @property
    def cells(self) -> Cells:
        """List of cells."""
        return self._cells

    @property
    def point_data(self) -> dict[str, Field]:
        """List of point data for the dataset."""
        return self._point_data

    @property
    def cell_data(self) -> dict[str, Field]:
        """List of cell data for the dataset."""
        return self._cell_data

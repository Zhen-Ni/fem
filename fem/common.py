#!/usr/bin/env python3

from __future__ import annotations
import abc
from enum import Enum
from dataclasses import dataclass
from typing import Generic, Type, TypeVar, overload, SupportsIndex, \
    SupportsFloat, Union
from collections.abc import Sequence, Iterator

try:
    # Available for sys.version > "3.11".
    from typing import Self
except ImportError:
    Self = TypeVar('Self')


__all__ = 'Vector', 'CellType', 'empty', 'XYZType'


@overload
def empty() -> None: ...


@overload
def empty(dim0: int, *dimensions: int) -> list: ...


def empty(*dimensions):
    """Create a multi-dimensional list filled with Nones."""
    if dimensions:
        return [empty(*dimensions[1:]) for i in range(dimensions[0])]
    return None


class CellType(Enum):
    VERTEX = 1
    LINE = 3
    QUAD = 9
    TETRA = 10
    HEXAHEDRON = 12


class Readonly:
    """Class with read-only attributes.

    All attributes in this class can only be assigned once.
    """
    __slots__ = ()

    def __setattr__(self, name: str, value):
        """Cannot modify existing attributes."""
        if hasattr(self, name):
            raise AttributeError(f"cannot modify '{name}' attribute of "
                                 f"read-only type '{self.__class__.__name__}'")
        return super().__setattr__(name, value)

    def __delattr__(self, name: str):
        raise AttributeError(f"cannot delete '{name}' attribute of "
                             f"read-only type '{self.__class__.__name__}'")


class InhertSlotsABCMeta(abc.ABCMeta):
    """A meta class to to build abstract class with slots.

    An empty __slots__ is defined in the constructed classes if the
    class doesn't define its own __slots__ attribute. Thus, all
    attributes in the constructed class should be declared in
    __slots__.
    """
    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        prepared_dict = super().__prepare__(name, bases, **kwargs)
        prepared_dict['__slots__'] = ()
        return prepared_dict


T = TypeVar('T')


class SequenceView(Sequence,
                   Readonly,
                   Generic[T],
                   metaclass=InhertSlotsABCMeta):
    """Get an immutable view object of given sequence."""

    __slots__ = '__sequence', '__indexes'

    def __init__(self,
                 sequence: Sequence[T],
                 indexes: Sequence[SupportsIndex] | slice | None = None):
        if isinstance(sequence, Sequence):
            self.__sequence = sequence
        else:
            raise TypeError('sequence must be instance of Sequence')
        # Make sure self.__indexes is a valid sequence of indexes.
        self.__indexes: Sequence[SupportsIndex]
        if indexes is None:
            self.__indexes = range(len(self.__sequence))
        elif isinstance(indexes, slice):
            # `Index out of range` never happends when using slice.
            self.__indexes = range(*indexes.indices(len(self.__sequence)))
        elif isinstance(indexes, Sequence):
            # Needs to check whether all indexes are in range.
            index_list = []
            for i in indexes:
                if not (-len(self.__sequence) <=
                        i.__index__() <
                        len(self.__sequence)):
                    raise IndexError('sequence index out of range')
                index_list.append(i)
            self.__indexes = index_list
        else:
            raise TypeError('indexes must be Sequence, slice or None,'
                            f' got {type(indexes)}')

    def __repr__(self) -> str:
        return '<View [' + ', '.join([repr(i) for i in self]) + ']>'

    @overload
    def __getitem__(self, index: SupportsIndex) -> T: ...

    @overload
    def __getitem__(self: Self,
                    index: slice | Sequence[SupportsIndex]) -> Self: ...

    def __getitem__(self, index):
        if isinstance(index, SupportsIndex):
            _index = index.__index__()
            if -len(self.__sequence) <= _index < len(self.__sequence):
                return self.__sequence[self.__indexes[index]]
            raise IndexError('index out of range')
        elif isinstance(index, (slice, Sequence)):
            # Check `index out of range` in __init__.
            return self.__class__(self, index)
        raise TypeError(
            f'{self.__class__.__name__} indices must be integers, slices, '
            f'or a sequence of integers,  not {type(index)}')

    def __len__(self) -> int:
        return len(self.__indexes)

    @property
    def _base_type(self) -> Type[Sequence]:
        if isinstance(self.__sequence, self.__class__):
            return self.__sequence._base_type
        return type(self.__sequence)

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self._base_type != other._base_type:
            return False
        if len(self) != len(other):
            return False
        for p1, p2 in zip(self, other):
            if p1 != p2:
                return False
        return True


XYZType = Union[tuple[()],
                tuple[SupportsFloat],
                tuple[SupportsFloat, SupportsFloat],
                tuple[SupportsFloat, SupportsFloat, SupportsFloat]
                ]


@dataclass(frozen=True, slots=True)
class _XYZBase:
    x: float = 0.
    y: float = 0.
    z: float = 0.


class XYZBase(_XYZBase, Readonly, metaclass=InhertSlotsABCMeta):
    """Base class for points and vectors in 3D space."""
    def __init__(self,
                 x: SupportsFloat = 0.,
                 y: SupportsFloat = 0.,
                 z: SupportsFloat = 0.):
        super().__init__(float(x), float(y), float(z))


class Vector(XYZBase):
    def normalize(self) -> Vector:
        return self / self.norm()

    def norm(self) -> float:
        return sum(i * i for i in self) ** .5

    def __repr__(self):
        return f"{self.__class__.__name__}: ({self.x}, {self.y}, {self.z})"

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def __mul__(self, c: float) -> Vector:
        return Vector(*(i * c for i in self))

    def __rmul__(self, c: float) -> Vector:
        return self * c

    def __truediv__(self, d: float) -> Vector:
        return self * (1. / d)

    def __neg__(self) -> Vector:
        return Vector(*(-i for i in self))

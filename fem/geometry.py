#!/usr/bin/env python3


from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Iterator, SupportsFloat, overload
from .common import Readonly, InhertSlotsABCMeta

__all__ = 'Point', 'Vector', 'ORIGIN', 'E1', 'E2', 'E3'


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


class Point(XYZBase):
    def __repr__(self):
        return f"Point: ({self.x}, {self.y}, {self.z})"

    def coordinates(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    @overload
    def __sub__(self, other: Point) -> Vector: ...

    @overload
    def __sub__(self, other: Vector) -> Point: ...

    def __sub__(self, other):
        if isinstance(other, Point):
            return Vector(self.x - other.x,
                          self.y - other.y,
                          self.z - other.z)
        if isinstance(other, Vector):
            return self + (-other)
        raise TypeError('unsupported operand type(s) for -: '
                        f'{type(self)} and {type(other)}')

    def __add__(self, other: Vector) -> Point:
        if isinstance(other, Vector):
            return Point(self.x + other.x,
                         self.y + other.y,
                         self.z + other.z)
        raise TypeError('unsupported operand type(s) for +: '
                        f'{type(self)} and {type(other)}')


class Vector(XYZBase):
    def normalize(self) -> Vector:
        return self / self.norm()

    def norm(self) -> float:
        return (self @ self) ** .5

    def __repr__(self):
        return f"{self.__class__.__name__}: ({self.x}, {self.y}, {self.z})"

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def __mul__(self, other: float) -> Vector:
        if isinstance(other, SupportsFloat):
            return Vector(*(i * other.__float__() for i in self))
        raise TypeError('unsupported operand type(s) for *: '
                        f'{type(self)} and {type(other)}')

    def __rmul__(self, other: float) -> Vector:
        if isinstance(other, SupportsFloat):
            return self * other.__float__()
        raise TypeError('unsupported operand type(s) for *: '
                        f'{type(self)} and {type(other)}')

    def __truediv__(self, other: float) -> Vector:
        if isinstance(other, SupportsFloat):
            return self * (1. / other.__float__())
        raise TypeError('unsupported operand type(s) for /: '
                        f'{type(self)} and {type(other)}')

    def __neg__(self) -> Vector:
        return Vector(*(-i for i in self))

    @overload
    def __add__(self, other: Vector) -> Vector: ...

    @overload
    def __add__(self, other: Point) -> Point: ...

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x,
                          self.y + other.y,
                          self.z + other.z)
        if isinstance(other, Point):
            return Point(self.x + other.x,
                         self.y + other.y,
                         self.z + other.z)
        raise TypeError('unsupported operand type(s) for +: '
                        f'{type(self)} and {type(other)}')

    def __sub__(self, other: Vector) -> Vector:
        if isinstance(other, Vector):
            return self + (-other)
        raise TypeError('unsupported operand type(s) for -: '
                        f'{type(self)} and {type(other)}')

    def __matmul__(self, other: Vector) -> float:
        if isinstance(other, Vector):
            return sum((a * b for (a, b) in zip(self, other)))
        raise TypeError('unsupported operand type(s) for @: '
                        f'{type(self)} and {type(other)}')

    def cross(self, other: Vector) -> Vector:
        return Vector(self.y * other.z - self.z * other.y,
                      self.z * other.x - self.x * other.z,
                      self.x * other.y - self.y * other.x)


ORIGIN = Point()
E1 = Vector(1, 0, 0)
E2 = Vector(0, 1, 0)
E3 = Vector(0, 0, 1)

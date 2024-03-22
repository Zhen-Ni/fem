#!/usr/bin/env python3

import re
from typing import TextIO, Literal, Type

from ..dataset import (Points, Cells, Point, Cell, Line, Quad, Hexahedron,
                       Mesh)


__all__ = 'read_inp',


class AbaqusInputFileError(Exception):
    pass


class AbaqusInp:
    """Interpreter of abaqus input file.

    This class can only process inp files with only one part at present. Users
    may use the "Create Mesh Part" function to generate the mesh into a new 
    model in the Abaqus CAE to generate this type of inp file. The nodes and
    elements of the model can then be extracted by using this class and 
    elements of different types can be seperated automatically.

    Parameters
    ----------
    stream : TextIO
        The input inp file stream.

    """

    # Match the line indicates the start of Node section, eg: *Node
    _IDENTIFIER_NODE = re.compile(r'\*node(?!\s*\w)', re.IGNORECASE)
    # Match the line indicates the start of Element section,
    # eg: *Element, type=B31
    _IDENTIFIER_ELEMENT = re.compile(r'\*element(?!\s*\w)', re.IGNORECASE)
    # Match the line indicates the start of comment section,
    # eg: ** This is comment of abaqus input file
    _IDENTIFIER_COMMENT = re.compile(r'\*\*', re.IGNORECASE)
    _IDENTIFIER = re.compile(r'\*\w', re.IGNORECASE)
    # Match the pattern definging element type.
    _PATTERN_TYPE = re.compile(r'(?<=type=)\w+', re.IGNORECASE)
    # Match floating point numbers.
    _PATTERN_FLOAT = re.compile(r'-?\d+(\.\d*)?(e-?\d+)?', re.IGNORECASE)
    # Match unsigned integers.
    _PATTERN_INTEGER = re.compile(r'\d+', re.IGNORECASE)

    def __init__(self, stream: TextIO):
        self._nodes: list[Point] = []
        self._elements: list[Cell] = []
        self._status: Literal['node', 'element', 'comment', 'unknown']
        self._status = 'unknown'
        # Use `None` for unknown types.
        self._element_type: Type[Cell] | None = None

        self._read(stream)

    def _read(self, stream: TextIO):
        cls = self.__class__
        self._status = 'unknown'
        for line_number, line_content in enumerate(stream):
            if re.match(cls._IDENTIFIER_NODE, line_content):
                self._status = 'node'
                continue
            elif re.match(cls._IDENTIFIER_ELEMENT, line_content):
                self._get_element_type(line_number, line_content)
                self._status = 'element'
                continue
            elif re.match(cls._IDENTIFIER_COMMENT, line_content):
                self._status = 'comment'
                continue
            elif re.match(cls._IDENTIFIER, line_content):
                self._status = 'unknown'
                continue

            match self._status:
                case 'node':
                    self._process_node(line_number, line_content)
                case 'element':
                    self._process_element(line_number, line_content)
                case _:
                    pass

    def _get_element_type(self, line_number: int, line_content: str):
        res = re.search(self.__class__._PATTERN_TYPE, line_content)
        if res is None:
            self._element_type = None
            raise AbaqusInputFileError('fail to identify element type at '
                                       f'line {line_number}:\n{line_content}')
        element_type = res.group()
        match element_type:
            case 'B31':
                self._element_type = Line
            case 'S4R' | 'S4':
                self._element_type = Quad
            case 'C3D8' | 'C3D8R' | 'C3D8S' | 'C3D8H' | 'C3D8I' | \
                 'C3D8RH' | 'C3D8IH' | 'C3D8HS':
                self._element_type = Hexahedron
            case _:
                self._element_type = Cell
                raise AbaqusInputFileError('unknown element type '
                                           f'{element_type} at line '
                                           f'{line_number}:'
                                           f'\n{line_content}')

    def _process_node(self, line_number: int, line_content: str):
        pattern = self.__class__._PATTERN_FLOAT
        words = [i.group() for i in re.finditer(pattern, line_content)]
        if len(words) != 4:
            raise AbaqusInputFileError("number of numbers in `*Node`"
                                       " section not correct at line "
                                       f"{line_number}:\n{line_content}")
        # We ignore the first number here as it is the index of the node.
        self._nodes.append(Point(*[float(i) for i in words[1:]]))

    def _process_element(self, line_number: int, line_content: str):
        pattern = self.__class__._PATTERN_INTEGER
        words = re.findall(pattern, line_content)
        if len(words) <= 1:
            raise AbaqusInputFileError("number of numbers in `*Element`"
                                       " section not correct at line "
                                       f"{line_number}:\n{line_content}")
        # An exception is raised when _element_type is unknown type at
        # _get_element_type.
        assert self._element_type is not None

        if len(words) - 1 != self._element_type.size:
            raise AbaqusInputFileError("number of numbers in `*Element`"
                                       " section not correct at line "
                                       f"{line_number}:\n{line_content}")
        # Note that indexes in abaqus starts from 1, but here we starts from 0.
        self._elements.append(self._element_type(
            *(int(i) - 1 for i in words[1:])))

    @property
    def mesh(self) -> Mesh:
        points = Points(self._nodes)
        cells = Cells(self._elements)
        return Mesh(points, cells)


def read_inp(filename: str) -> Mesh:
    with open(filename) as f:
        ds = AbaqusInp(f).mesh
    return ds

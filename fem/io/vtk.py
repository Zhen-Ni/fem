#!/usr/bin/env python3

from __future__ import annotations

from typing import TextIO, Literal

import vtk

from ..dataset import Mesh, Dataset, Field, \
    FloatScalarField, ComplexScalarField, \
    FloatArrayField, ComplexArrayField


__all__ = 'write_vtk', 'write_vtu'


def write_vtu(stream: TextIO,
              dataset: Dataset,
              data_mode: Literal['ascii', 'binary', 'appended'] = 'ascii',
              compressor: Literal['none', 'lz4', 'zlib', 'lzma'] = 'none'
              ) -> int:
    """Write dataset into VTK XML UnstructuredGrid (.vtu) format.

    Parameters
    ----------
    stream : TextIO
        Opened file stream to be written into.
    dataset : DataSet object
        Dataset to be written.
    """
    vtk_dataset = _dataset_to_vtu(dataset)
    writer = vtk.vtkXMLUnstructuredGridWriter()

    match data_mode.lower():
        case 'ascii':
            writer.SetDataMode(writer.Ascii)
        case 'binary':
            writer.SetDataMode(writer.Binary)
        case 'appended':
            writer.SetDataMode(writer.Appended)
        case _:
            raise ValueError('filetype shoud be "ascii", '
                             '"binary" or "appended", '
                             f'got"{data_mode}"')

    match compressor.lower():
        case 'none':
            writer.SetCompressorTypeToNone()
        case 'lz4':
            writer.SetCompressorTypeToLZ4()
        case 'zlib':
            writer.SetCompressorTypeToZLib()
        case 'lzma':
            writer.SetCompressorTypeToLZMZ()
        case _:
            raise ValueError('filetype shoud be "none", '
                             '"lz4", "zlib" or "lzma", '
                             f'got"{data_mode}"')

    writer.WriteToOutputStringOn()

    writer.SetInputData(vtk_dataset)
    writer.Write()
    string = writer.GetOutputString()
    return stream.write(string)


def _dataset_to_vtu(dataset: Dataset) -> vtk.vtkUnstructuredGrid:
    """Convert dataset to vtkUnstructuredGrid object.

    Parameters
    ----------
    stream : TextIO
        Opened file stream to be written into.
    dataset : DataSet object
        Dataset to be written.
    """
    vtk_dataset = vtk.vtkUnstructuredGrid()

    # Write points
    points = vtk.vtkPoints()
    for i, p in enumerate(dataset.mesh.points):
        points.InsertPoint(i, [p.x, p.y, p.z])
        vtk_dataset.SetPoints(points)

    # Write cells
    vtk_dataset.Allocate(len(dataset.mesh.cells))
    for i, c in enumerate(dataset.mesh.cells):
        point_ids = c.nodes
        vtk_dataset.InsertNextCell(c.id.value, c.size, point_ids)

    # Write point data
    for (key, field) in dataset.point_data.items():
        for array in _vtu_convert_field(key, field):
            vtk_dataset.GetPointData().AddArray(array)

    # Write cell data
    for (key, field) in dataset.cell_data.items():
        for array in _vtu_convert_field(key, field):
            vtk_dataset.GetCellData().AddArray(array)

    return vtk_dataset


def _vtu_convert_field(key: str,
                       field: Field
                       ) -> list[vtk.vtkDoubleArray]:
    """Convert fields to vtkDoubleArray.

    For field with floating point data, a list of a single vtk array
    is returned. For field with complex data, a list of two vtk arrays
    corresponding to the real and imaginary parts of the data is
    returned.

    This is a helper function of `_dataset_to_vtk`.
    """
    # scalar field
    if isinstance(field, FloatScalarField):
        return [_vtu_convert_floatarrayfield(key, field.to_array_field())]
    elif isinstance(field, ComplexScalarField):
        _field = field.to_array_field()
        return [_vtu_convert_floatarrayfield(key + ': real', _field.real()),
                _vtu_convert_floatarrayfield(key + ': imag', _field.imag())]
    # vector field
    elif isinstance(field, FloatArrayField):
        return [_vtu_convert_floatarrayfield(key, field)]
    elif isinstance(field, ComplexArrayField):
        return [_vtu_convert_floatarrayfield(key + ': real', field.real()),
                _vtu_convert_floatarrayfield(key + ': imag', field.imag())]
    raise TypeError(f'unsupported field type: {type(field)}')


def _vtu_convert_floatarrayfield(key: str,
                                 field: FloatArrayField
                                 ) -> vtk.vtkDoubleArray:
    """Convert float array fields to vtkDoubleArray.

    This is a helper function of `_vtu_convert_field`.
    """
    array = vtk.vtkDoubleArray()
    array.SetNumberOfComponents(field.dimension)
    array.SetNumberOfTuples(len(field))
    array.SetName(key)
    for i, value in enumerate(field):
        array.SetTuple(i, value)
    return array


def write_vtk(stream: TextIO,
              dataset: Dataset) -> int:
    """Write dataset into legency VTK format.

    Parameters
    ----------
    stream : TextIO
        Opened file stream to be written into.
    dataset : DataSet object
        Dataset to be written.
    """
    content = _dataset_to_vtk(dataset)
    return stream.write(content)


def _dataset_to_vtk(dataset: Dataset) -> str:
    """Convert given dataset into legency VTK format.

    Note that for Field information with complex data, the real and
    imaginary parts will be stored seperately.

    Parameters
    ----------
    dataset : Dataset object
        Dataset to be converted.

    Returns
    -------
    string
        Dataset in legency VTK format.

    Reference
    ---------
    [1] The VTK User's Guide (11th Edition).
        https://vtk.org/wp-content/uploads/2021/08/VTKUsersGuide.pdf.
    """
    lines = []

    # Part1: Header
    lines.append('# vtk DataFile Version 3.0')

    # Part2: Title
    # The maximum length of the title if 256 characters, including the
    # ending '\n'.
    if len(dataset.title) < 256:
        lines.append(dataset.title)
    else:
        raise ValueError('length of title of the dataset should be '
                         'less than 256 for converting to '
                         'legency VTK format.')

    # Part3: Data type, either ASCII or BINARY
    # We use ASCII here.
    lines.append('ASCII')

    # Part 4: Geometry/topology
    # Dataset stores only unstructured grids.
    mesh_lines = _vtk_convert_mesh(dataset.mesh)
    lines.extend(mesh_lines)

    # Part 5: dataset attributes
    # write point data
    lines.append('POINT_DATA {}'.format(len(dataset.mesh.points)))
    for key, field in dataset.point_data.items():
        attr_lines = _vtk_convert_field(key, field)
        lines.extend(attr_lines)

    # write cell data
    lines.append('CELL_DATA {}'.format(len(dataset.mesh.cells)))
    for key, field in dataset.cell_data.items():
        attr_lines = _vtk_convert_field(key, field)
        lines.extend(attr_lines)

    lines.append('')
    return '\n'.join(lines)


def _vtk_convert_mesh(mesh: Mesh) -> list[str]:
    """Convert mesh to part 4 of legency vtk file.

    Lines to be written to part 4 of the file are collected and
    returned in a list.

    This is a helper function of `_dataset_to_vtk`.
    """
    lines = []
    # Dataset stores only unstructured grids.
    lines.append('DATASET UNSTRUCTURED_GRID')
    # write the points
    lines.append('POINTS {} double'.format(len(mesh.points)))
    for point in mesh.points:
        lines.append('{} {} {}'.format(*point.coordinates))
    # write the cells
    size = sum([c.size + 1 for c in mesh.cells])
    lines.append('CELLS {} {}'.format(len(mesh.cells), size))
    for cell in mesh.cells:
        lines.append(' '.join(['{}'.format(cell.size)] +
                              ['{}'.format(n) for n in cell.nodes]))
    # cell types
    lines.append('CELL_TYPES {}'.format(len(mesh.cells)))
    for cell in mesh.cells:
        lines.append(f'{cell.id.value}')
    return lines


def _vtk_convert_field(key: str, field: Field, ) -> list[str]:
    """Convert fields to dataset attributes in part 5 of legency vtk
    file.

    Lines to be written to part 5 of the file are collected and
    returned in a list.

    This is a helper function of `_dataset_to_vtk`.
    """
    lines = []
    # scalar field
    if isinstance(field, FloatScalarField):
        attr_lines = _vtk_convert_scalarfield(key, field)
        lines.extend(attr_lines)
    elif isinstance(field, ComplexScalarField):
        attr_lines = _vtk_convert_scalarfield(key + ': real', field.real())
        lines.extend(attr_lines)
        attr_lines = _vtk_convert_scalarfield(key + ': imag', field.imag())
        lines.extend(attr_lines)
    # vector field
    elif isinstance(field, FloatArrayField):
        attr_lines = _vtk_convert_arrayfield(key, field)
        lines.extend(attr_lines)
    elif isinstance(field, ComplexArrayField):
        attr_lines = _vtk_convert_arrayfield(key + ': real', field.real())
        lines.extend(attr_lines)
        attr_lines = _vtk_convert_arrayfield(key + ': imag', field.imag())
        lines.extend(attr_lines)
    else:
        raise TypeError(f'unsupported field type: {type(field)}')
    return lines


def _vtk_convert_scalarfield(key: str,
                             field: FloatScalarField) -> list[str]:
    """Convert scalarfield to point data in part 5 of legency vtk file.

    Lines to be written to part 5 of the file are collected and
    returned in a list.

    This is a helper function of `_vtk_convert_field`
    """
    lines = []
    lines.append('SCALARS {} double'.format(key))
    lines.append('LOOKUP_TABLE default')
    for scalar_value in field:
        lines.append('{}'.format(scalar_value))
    return lines


def _vtk_convert_arrayfield(key: str,
                            field: FloatArrayField) -> list[str]:
    """Convert arrayfield to field data in part 5 of legency vtk file.

    Lines to be written to part 5 of the file are collected and
    returned in a list.

    This is a helper function of `_vtk_convert_field`
    """
    lines = []
    lines.append('FIELDS {} 1'.format(key))
    lines.append('{} {} {} double'.format(key,
                                          field.dimension, len(field)))
    for array in field:
        lines.append(' '.join(['{}'.format(i) for i in array]))
    return lines

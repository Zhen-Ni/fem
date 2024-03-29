#!/usr/bin/env python3

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
import os
import xml.etree.ElementTree as ET

from .vtk import write_vtu

if TYPE_CHECKING:
    from ..dataset import Dataset

__all__ = 'export_pvd',


def export_pvd(datasets: Sequence[Dataset], name: str, path: str = ''):
    """Write a list of datasets to paraview .pvd file.

    Paraview's .pvd file format is a pointer to a collection of VTK
    XML files, with each file corresponding a timestep. Thus, a .pvd
    file and a folder containing the vtk files are created.

    Parameters
    ----------
    datasets : a sequence of DataSet objects
        Datasets to be written.
    name : str
        The name of the files. A `name`.pvd file and a folder named
        `name containing corresponding vtk files will be created
    path : str, optional
        Path of the files.
    """
    # Create .pvd file
    pvd_xml = ET.Element("VTKFile",
                         attrib={'type': 'Collection',
                                 'byte_order': 'LittleEndian',
                                 'compressor': 'vtkZLibDataCompressor'})
    collection = ET.SubElement(pvd_xml, 'Collection')
    file_prefix = os.path.join(name, name)
    for i, dataset in enumerate(datasets):
        ET.SubElement(collection, 'DataSet',
                      attrib={'timestep': '{}'.format(dataset.time),
                              'part': '0',
                              'file': file_prefix + '-{}.vtu'.format(i)})
    et = ET.ElementTree(pvd_xml)
    et.write(os.path.join(path, name)+'.pvd',
             encoding="utf-8",
             xml_declaration=True)

    # Create the folder and its contents.
    os.makedirs(name, exist_ok=True)
    for i, d in enumerate(datasets):
        with open(os.path.join(path, name, f'{name}-{i}.vtu'), 'w') as f:
            write_vtu(f, d, 'appended', 'zlib')

#!/usr/bin/env python3

"""dataset provides a interface for transfer data between different programs 
by using the VTK file format.
"""

import os
import os.path
import struct
import base64
import zlib
import xml.etree.ElementTree as ET


__all__ = ['Point', 'Cell', 'Field', 'ScalarField', 'VectorField', 'DataSet',
           'dumppvd','dumpvtu', 'dumpvtk']


VTK_VERTEX = 1
VTK_LINE = 3
VTK_QUAD = 9

class Point():
    """Point class for Dataset object.
    
    The instance of the Point class represents a point in the 3-D space.
    
    Parameters
    ----------
    *args: arguments
        The `Point` class can be instantiated with 1 or 3 arguments. The 
        following gives the number of input arguments and their interpretation:
            
            * 1: iterable with three elements representing the coordinate of
                 the x, y and z axis
            * 3: coordinate of the point (x, y and z)
    """
    
    def __init__(self, *args):
        try:
            if len(args) == 1:
                x, y, z = args[0]
            else:
                x, y, z = args
        except ValueError:
            raise ValueError('Point() takes 1 or 3 positional arguments')
        self._coordinate = (x, y, z)
    
    def __repr__(self):
        return "Point: {coor}".format(coor=self._coordinate)
    
    @property
    def coordinate(self):
        return self._coordinate
    @property
    def x(self):
        return self._coordinate[0]
    @property
    def y(self):
        return self._coordinate[1]
    @property
    def z(self):
        return self._coordinate[2]

class Cell:
    """Cell class for Dataset object.
    
    The object of the Cell class represents a cell (or element). It contains
    information of the cell type and the node indexes of the cell. Users may
    refer to VTK file formats for information about the cell types.
    
    Paramteres
    ----------
    points: iterable
        List of the indexes of vertices (Point instances) that forms the cell.
    cell_type: int
        Type of the cell. Users may refer to VTK file formats for more
        information.
    """
    def __init__(self, points, cell_type):
        self._points = tuple(points)
        self._cell_type = int(cell_type)
    
    def __repr__(self):
        return ("Cell: {points}, cell_type = {cell_type}"
                .format(points=self._points, cell_type=self._cell_type))
    
    def cell_size(self):
        """Return the number of vertics that forms the cell."""
        return len(self._points)
    
    @property
    def points(self):
        return self._points
    
    @property
    def cell_type(self):
        return self._cell_type
    

class Field:
    """Base class for field data in the Dataset.
    
    An instance of the Field class defined a named field for point data or cell
    data. The field data can should be iterable with a constant length. For 
    scalar field data, users may use ScalarField instead.
    
    Parameters
    ----------
    data_name: string
        Name of the field.
    data: iterable
        Data for each point or cell.
    ncomponents: int
        Length of the field data for each point or cell.
    
    See Alse
    --------
    ScalarField, VectorField
    """
    def __init__(self, data_name, data, ncomponents):
        self._data_name = data_name
        self._data = tuple(data)
        self._ncomponents = int(ncomponents)

    def __repr__(self):
        return ("Field: {name}".format(name=self._data_name))

    def size(self):
        return len(self._data)

    @property
    def data_name(self):
        return self._data_name
    
    @property
    def data(self):
        return self._data


class ScalarField(Field):
    """Class for field data in the Dataset.
    
    An instance of the Field class defined a named field for point data or cell
    data. The field data can should be scalar.
    
    Parameters
    ----------
    data_name: string
        Name of the field.
    scalar: iterable
        Data for each point or cell.

    See Alse
    --------
    VectorField
    """
    def __init__(self, data_name, scalar):
        super().__init__(data_name, scalar, -1)

    def __repr__(self):
        return ("Scalar Field: {name}".format(name=self._data_name))
    
    @property
    def scalars(self):
        return self._data

class VectorField(Field):
    """Class for field data in the Dataset.
    
    An instance of the Field class defined a named field for point data or cell
    data. The field data can should be iterable with a constant length. For 
    scalar field data, users may use ScalarField instead.
    
    Parameters
    ----------
    data_name: string
        Name of the field.
    data: iterable
        Data for each point or cell.
    ncomponents: int, optional
        Length of the field data for each point or cell. (default to 3)
    
    See Alse
    --------
    ScalarField
    """
    def __init__(self, data_name, vectors, ncomponents=3):
        super().__init__(data_name, vectors, ncomponents)

    def __repr__(self):
        return ("Vector Field: {name}".format(name=self._data_name))
    
    @property
    def vectors(self):
        return self._data

    @property
    def ncomponents(self):
        return self._ncomponents
    

class DataSet:
    """Class for mesh geometry and the corresponding field data.
    
    DataSet class corresponds to VTK file formats using unstructured mesh. It 
    stores the points and cells to construct the geometry. Field data can be
    added to the instances to represent the point data or cell data.
    
    Parameters
    ----------
    points: iterable
        Iterable object containing Point instances.  
    cells: iterable
        Iterable object containing Cell instances.
    title: string, optional
        The title for the VTK file. Should be no longer than 255 characters.
    time: float, optional
        Additional information for time step.
    """
    def __init__(self, points=None, cells=None, title='', time=None):
        self._title = ''
        self._time = time
        self._points = []
        self._cells = []
        self._point_data = {}
        self._cell_data = {}
        
        self.set_title(title)
        if points is not None:
            self.points.extend(points)
        if cells is not None:
            self.cells.extend(cells)   
    
    def set_title(self, title):
        """Set the title for the dataset."""
        title = str(title)
        if len(title) > 255:
            raise ValueError('title should be no longer than 255 characters')
        self._title = title
    
    @property
    def title(self):
        return self._title
    
    @property
    def time(self):
        return self._time
    
    @time.setter
    def time(self, time):
        self._time = time
    
    @property
    def points(self):
        """List of points."""
        return self._points
    
    @property
    def cells(self):
        """List of cells."""
        return self._cells
    
    @property
    def point_data(self):
        """List of point data for the dataset."""
        return self._point_data
    
    @property
    def cell_data(self):
        """List of cell data for the dataset."""
        return self._cell_data



def dumppvd(datasets, filename, path=None):
    """Write the datasets into pvd format. (paraview data format)
    
    A file named "`filename`.pvd" and its corresponding folder will be written
    to the disk in `path`.
    
    Parameters
    ----------
    datasets: DataSet object or a list of DataSet objects
        Dataset to be written.
    
    filename: string
        File to be written. If filename is None, the pvd file will not be
        written to the disk.
    
    path: string, optional
        Path of the file. (default to None)
    
    Returns
    -------
    pvd: Element
        Element containing information in pvd file.
    
    Side Effects
    ------------
    Write files to the disk.
    """
    try:
        iter(datasets)
    except TypeError:
        datasets = [datasets]
    if path is None:
        path = ''
    try:
        os.mkdir(os.path.join(path,filename))
    except FileExistsError:
        pass
    dumpvtu(datasets, filename, os.path.join(path, filename))
    pvd_xml = _dumppvd_helper(datasets, filename, path)
    return pvd_xml
    
        

def _dumppvd_helper(datasets, filename, path):
    """Write the pvd file.
        
    Parameters
    ----------
    datasets: a list of DataSet objects
        Dataset to be written.

    Returns
    -------
    pvd: Element
        Element containing information in pvd file.
    """
    pvd_xml = ET.Element("VTKFile", attrib={'type':'Collection',
                                            'byte_order':'LittleEndian'})
    collection = ET.SubElement(pvd_xml, 'Collection')
    file_prefix = os.path.join(filename,filename)
    for i, dataset in enumerate(datasets):
        ds = ET.SubElement(collection, 'DataSet',
                           attrib={'timestep':'{}'.format(dataset.time),
                                   'part':'0',
                                   'file':file_prefix+'-{}.vtu'.format(i)})
        
    et = ET.ElementTree(pvd_xml) #生成文档对象
    et.write(os.path.join(path, filename)+'.pvd',
             encoding="utf-8",xml_declaration=True)
    return pvd_xml

def dumpvtu(dataset, filename=None, path=None, compress=True):
    """Write the dataset into VTK XML format.
    
    If `dataset` is a DataSet object, a string will be returned, and a file
    named "`filename`.vtu" will be written to the disk if `filename` is not
    None. if `dataset` is a list of DataSet objects, a list of string will be
    returned, and a list of file named "`filename`-i.vtu" will be written to
    the disk if `filename` is not None where i is the index of the 
    corresponding dataset.
    
    Parameters
    ----------
    dataset: DataSet object or a list of DataSet objects
        Dataset to be written.
    filename: string, optional
        File to be written. If filename is None, the vtk file will not be
        written to the disk. (default to None)
    path: string, optional
        Path of the file. (default to None)
    compress: bool, optional
        Whether to compress binary data. (default to True)
    
    Returns
    -------
    vtu_xml: Element
        Element containing information in `dataset`.
    
    Side Effects
    ------------
    Write a file or a list of files to the disk if `filename` is not None
    """
    if filename is not None and path is not None:
        filename = os.path.join(path, filename)
    try:
        res = []
        for i,ds in enumerate(dataset):
            res.append(_dumpvtu_helper(ds, filename+'-{i}'.format(i=i),
                                       compress))
    except TypeError:
        res = _dumpvtu_helper(dataset, filename, compress)
    return res


def _dumpvtu_helper(dataset, filename, compress):
    """Process only one dataset."""
    vtu_xml = _dumpvtu_dumper(dataset, compress)
    if filename is not None:
        et = ET.ElementTree(vtu_xml) #生成文档对象
        et.write(filename+'.vtu', encoding="utf-8",xml_declaration=True)
    return vtu_xml
        

def _dumpvtu_dumper(dataset, compress):
    """Write the dataset into XML format.
    
    Note that only real part of the field data will be written to the output.
    
    Parameters
    ----------
    dataset: DataSet object
        Dataset to be written.
    compress: bool
        Whether to compress binary data.
    
    Returns
    -------
    vtu_xml: Element
        Element containing information in `dataset`.
    """
    appended_data = bytearray()
    vtu_xml = ET.Element("VTKFile", attrib={'type':'UnstructuredGrid',
                                            'byte_order':'LittleEndian'})
    if compress:
        vtu_xml.set('compressor','vtkZLibDataCompressor')
        _pack_list = _pack_list_compressed
    else:
        _pack_list = _pack_list_plain
        
    unstructuredgrid = ET.SubElement(vtu_xml, 'UnstructuredGrid')
    piece = ET.SubElement(unstructuredgrid, 'Piece',
              attrib={'NumberOfPoints':'{:d}'.format(len(dataset.points)),
                      'NumberOfCells':'{:d}'.format(len(dataset.cells))})
    
    # the order of the elements in `piece`: PointData, CellData, Points, Cells
    
    # PointData
    pointdata = ET.SubElement(piece, 'PointData')
    for key,field in dataset.point_data.items():
        dataarray = ET.SubElement(pointdata, 'DataArray',
                                  attrib={'Name':field.data_name,
                                          'type':'Float64',
                                          'format':'appended',
                                          'offset':'{:d}'.format(len(appended_data))})
        # scalars
        if type(field) == ScalarField:
            data = [i.real for i in field.data]
        # vectors
        else:
            dataarray.set('NumberOfComponents','{:d}'.format(field.ncomponents))
            data = []
            [data.extend([i.real for i in d]) for d in field.data]
        appended_data.extend(_pack_list('<d', data))
        
    # CellData
    celldata = ET.SubElement(piece, 'CellData')
    for key,field in dataset.cell_data.items():
        dataarray = ET.SubElement(celldata, 'DataArray',
                                  attrib={'Name':field.data_name,
                                          'type':'Float64',
                                          'format':'appended',
                                          'offset':'{:d}'.format(len(appended_data))})
        # scalars
        if type(field) == ScalarField:
            data = [i.real for i in field.data]
        # vectors
        else:
            dataarray.set('NumberOfComponents','{:d}'.format(field.ncomponents))
            data = []
            [data.extend([i.real for i in d]) for d in field.data]
        appended_data.extend(_pack_list('<d', data))
        
    # Points
    points = ET.SubElement(piece, 'Points')
    dataarray = ET.SubElement(points, 'DataArray',
                              attrib={'type':'Float64',
                                      'NumberOfComponents':'3',
                                      'format':'appended',
                                      'offset':'{:d}'.format(len(appended_data))})
    data = []
    [data.extend(d.coordinate) for d in dataset.points]
    appended_data.extend(_pack_list('<d', data))
    
    # Cells
    # Cells contain three elements: connectivity, offsets and types
    cells = ET.SubElement(piece, 'Cells')
    # conncectivity
    dataarray = ET.SubElement(cells, 'DataArray',
                              attrib={'type':'Int32',
                                      'Name':'connectivity',
                                      'format':'appended',
                                      'offset':'{:d}'.format(len(appended_data))})
    data = []
    [data.extend(p) for p in [c.points for c in dataset.cells]]
    appended_data.extend(_pack_list('<i', data))
    # offsets
    dataarray = ET.SubElement(cells, 'DataArray',
                              attrib={'type':'Int32',
                                      'Name':'offsets',
                                      'format':'appended',
                                      'offset':'{:d}'.format(len(appended_data))})
    data = []
    offset = 0
    for c in dataset.cells:
        offset += len(c.points)
        data.append(offset)
    appended_data.extend(_pack_list('<i', data))
    # types
    dataarray = ET.SubElement(cells, 'DataArray',
                              attrib={'type':'UInt8',
                                      'Name':'types',
                                      'format':'appended',
                                      'offset':'{:d}'.format(len(appended_data))})
    data = [c.cell_type for c in dataset.cells]
    appended_data.extend(_pack_list('<B', data))
    # Appended data
    ET.SubElement(vtu_xml, 'AppendedData',
                  attrib={'encoding':'base64'}).text = '_' + appended_data.decode()
    
    return vtu_xml
    
def _pack_list_plain(fmt, data):
    """Pack data into binary form and encode it using base64.
    
    The resulting data contains two parts. The first part is a 32-bit interger
    containg the length of the packed data (in bytes), The second part is the
    packed data. The two parts are encoded seperately.
    
    The pseudo code is as follows:  
        
    int32 = length( data );  
    output = base64-encode( int32 ) + base64-encode( data );
    
    Parameters
    ----------
    fmt: string
        Format of the binary data. See struct for more details.
    data: list
        New data to be packed.
    
    Returns
    -------
    bdata: bytearray
        Packed data.
    """
    bdata2 = bytearray()    # data
    for d in data:
        bdata2.extend(struct.pack(fmt,d))
    bdata1 = struct.pack('<i',len(bdata2))   # length of data
    bdata1 = base64.encodebytes(bdata1)
    bdata2 = base64.encodebytes(bdata2)
    bdata = bdata1 + bdata2
    bdata = b''.join(bdata.split(b'\n'))
    return bdata


def _pack_list_compressed(fmt, data, level=-1):
    """Pack data into compressed binary form and encode it using base64.
    
    The resulting data contains two parts. The first part is a 32-bit interger
    array (header). The second part is the packed data. The two parts are
    encoded seperately.
    
    The pseudo code is as follows:  
        
    int32[0] = 1;
    int32[1] = length( data ); 
    int32[2] = length( data ); 
    zdata = compress( data );
    int32[3] = length( zdata );
    output = base64-encode( int32 ) + base64-encode( zdata );
    
    Parameters
    ----------
    fmt: string
        Format of the binary data. See struct for more details.
    data: list
        New data to be packed.
    level: int (optional)
        Compression level, in 0-9.
    Returns
    -------
    bdata: bytearray
        Packed data.
    """
    bdata2 = bytearray()    # data
    for d in data:
        bdata2.extend(struct.pack(fmt,d))
    bdata1 = bytearray()   # header
    bdata1.extend(struct.pack('<i',1))
    bdata1.extend(struct.pack('<i',len(bdata2)))
    bdata1.extend(struct.pack('<i',len(bdata2)))
    bdata2 = zlib.compress(bdata2)
    bdata1.extend(struct.pack('<i',len(bdata2)))
    bdata1 = base64.encodebytes(bdata1)
    bdata2 = base64.encodebytes(bdata2)
    bdata = bdata1 + bdata2
    bdata = b''.join(bdata.split(b'\n'))
    return bdata


def dumpvtk(dataset, filename=None, path=None):
    """Write the dataset into legency VTK format.
    
    If `dataset` is a DataSet object, a string will be returned, and a file
    named "`filename`.vtk" will be written to the disk if `filename` is not
    None. if `dataset` is a list of DataSet objects, a list of string will be
    returned, and a list of file named "`filename`-i.vtk" will be written to
    the disk if `filename` is not None where i is the index of the 
    corresponding dataset.
    
    Parameters
    ----------
    dataset: DataSet object or a list of DataSet objects
        Dataset to be written.
    
    filename: string, optional
        File to be written. If filename is None, the vtk file will not be
        written to the disk. (default to None)
    
    path: string, optional
        Path of the file. (default to None)
    
    Returns
    -------
    slf: string or a list of strings
        Dataset in legency VTK format.
    
    Side Effects
    ------------
    Write a file or a list of files to the disk if `filename` is not None
    """
    if filename is not None and path is not None:
        filename = os.path.join(path, filename)
    try:
        res = []
        for i,ds in enumerate(dataset):
            res.append(_dumpvtk_helper(ds, filename+'-{i}'.format(i=i)))
    except TypeError:
        res = _dumpvtk_helper(dataset, filename)
    return res


def _dumpvtk_helper(dataset, filename):
    """Process only one dataset."""
    slf = _dumpvtk_dumper(dataset)
    if filename is not None:
        with open(filename+'.vtk', 'w') as file:
            file.write(slf)
    return slf
            
            
def _dumpvtk_dumper(dataset):
    """Write the dataset into legency VTK format.
    
    Note that only real part of the field data will be written to the output.
    
    Parameters
    ----------
    dataset: DataSet object
        Dataset to be written.
    
    Returns
    -------
    slf: string
        Dataset in legency VTK format.
    """
    slf = []
    # write the head
    slf.append('# vtk DataFile Version 3.0')
    slf.append(dataset.title)
    slf.append('ASCII')
    slf.append('DATASET UNSTRUCTURED_GRID')
    # write the points
    slf.append('POINTS {} double'.format(len(dataset.points)))
    for point in dataset.points:
        slf.append('{} {} {}'.format(*point.coordinate))
    # write the cells
    size = sum([c.cell_size()+1 for c in dataset.cells])
    slf.append('CELLS {} {}'.format(len(dataset.cells), size))
    for cell in dataset.cells:
        slf.append(' '.join(['{:d}'.format(cell.cell_size())] +
                            ['{:d}'.format(p) for p in cell.points]))
    
    slf.append('CELL_TYPES {}'.format(len(dataset.cells)))
    for cell in dataset.cells:
        slf.append('{:d}'.format(cell.cell_type))
    # write point data
    slf.append('POINT_DATA {}'.format(len(dataset.points)))
    for key,field in dataset.point_data.items():
        # scalars
        if type(field) == ScalarField:
            slf.append('SCALARS {} double'.format(field.data_name))
            slf.append('LOOKUP_TABLE default')
            for d in field.data:
                slf.append('{}'.format(d.real))
###############################################################################
#        ## Deprecated                                                        #
#        # vectors                                                            #
#        elif type(field) == VectorField:                                     #
#            slf.append('VECTORS {} double'.format(field.data_name))          #
#            for d in field.data:                                             #
#                slf.append('{} {} {}'.format(*d))                            #
###############################################################################
        # vectors (VectorField or Field), use field expression in VTK
        else:
            slf.append('FIELDS {} 1'.format(key))
            slf.append('{} {} {} double'.format(field.data_name,
                       field.ncomponents, field.size()))
            for d in field.data:
                slf.append(' '.join(['{}'.format(i.real) for i in d]))
    # write cell data
    slf.append('CELL_DATA {}'.format(len(dataset.cells)))
    for key,field in dataset.cell_data.items():
        # scalars
        if type(field) == ScalarField:
            slf.append('SCALARS {} double'.format(field.data_name))
            slf.append('LOOKUP_TABLE default')
            for d in field.data:
                slf.append('{}'.format(d.real))
###############################################################################
#        ## Deprecated                                                        #
#        # vectors                                                            #
#        elif type(field) == VectorField:                                     #
#            slf.append('VECTORS {} double'.format(field.data_name))          #
#            for d in field.data:                                             #
#                slf.append('{} {} {}'.format(*d))                            #
###############################################################################
        # vectors (VectorField or Field), use field expression in VTK
        else:
            slf.append('FIELDS {} 1'.format(key))
            slf.append('{} {} {} double'.format(field.data_name,
                       field.ncomponents, field.size()))
            for d in field.data:
                slf.append(' '.join(['{}'.format(i.real) for i in d]))
    slf.append('')
    return '\n'.join(slf)
    
    
    
if __name__ == '__main__':    
    points = [(Point(0,0,0)),(Point(1,0,0)),(Point(2,0,0)),
              (Point(0,1,0)),(Point(1,1,0)),(Point(2,1,0)),
              (Point(1,2,0)),(Point(2,2,0))]
    cells = [Cell([0,1,4,3],VTK_QUAD),Cell([1,2,5,4],VTK_QUAD),
                  Cell([4,5,7,6],VTK_QUAD)]
    
    temp = ScalarField('temprature', [0,1,2,1,2,3,3,4])
    disp = VectorField('displacement',[[0,0,0],[0,0,1],[0,0,0],
                                       [0,0,1],[0,0,2],[0,0,1],
                                       [0,0,1],[0,0,0]])
    coor = VectorField('coordinate', [[0,0],[1,0],[2,0],
                                      [0,1],[1,1],[2,1],
                                      [1,2],[2,2]], 2)
    dataset = DataSet(points, cells, 'test')
    dataset.point_data['temp'] = temp
    dataset.point_data['disp'] = disp
    dataset.point_data['coor'] = coor
    
    density = ScalarField('density', [1,1.5,2])
    dataset.cell_data['rho'] = density
    
    slf = dumpvtk(dataset)
    with open('dataset-test.vtk','w') as f:
        f.write(slf)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
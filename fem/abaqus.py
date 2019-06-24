#!/usr/bin/env python3

import sys
import re

import numpy as np
from scipy.sparse import coo_matrix
from .dataset import *


__all__ = ['AbaqusInputFile', 'read_matrix_coordinate']


class AbaqusInputFile:
    """Interpreter of abaqus input file.
    
    This class can only process inp files with only one part at present. Users
    may use the "Create Mesh Part" function to generate the mesh into a new 
    model in the Abaqus CAE to generate this type of inp file. The nodes and
    elements of the model can then be extracted by using this class and 
    elements of different types can be seperated automatically.
    
    Parameters
    ----------
    stream: stream
        The input inp file stream.
    
    """
    
    
    def __init__(self, stream):
        self._nodes = []
        self._element_types = []
        self._elements = []

        self._identifier_node = re.compile('\*node(?!\s*\w)',re.IGNORECASE)
        self._identifier_element = re.compile('\*element(?!\s*\w)',re.IGNORECASE)
        self._identifier_comment = re.compile('\*\*',re.IGNORECASE)
        self._identifier = re.compile('\*\w',re.IGNORECASE)
        self._pattern_type = re.compile('(?<=type=)\w+',re.IGNORECASE)
        self._pattern_digit = re.compile('-?\d+(\.\d*)?(e-?\d+)?',re.IGNORECASE)
        self._pattern_integer = re.compile('\d+',re.IGNORECASE)

        self._read(stream)

        self._nodes = np.array(self._nodes, dtype=float)
        self._elements = [np.array(i, dtype=int)-1 for i in self._elements]

    def _read(self, stream):
        status = 'unknown'
        line_number = 0
        for line in stream:
            line_number += 1
            # print(line_number, line,sep='\t', end='')
            if re.match(self._identifier_node,line):
                status = 'node'
                continue
            if re.match(self._identifier_element,line):
                status = 'element'
                self._process_element_identifier(line)
                continue
            if re.match(self._identifier,line):
                status = 'unknown'
                continue
            if re.match(self._identifier_comment,line):
                continue

            try:
                self._process_line(line, status)
            except Exception:
                sys.displayhook(sys.exc_info())
                sys.stderr.write('Error when processing line {line_number}\n'
                                 .format(line_number=line_number))
                sys.stderr.write(line+'\n')
                res = input('Continue?(Y/N)')
                if res.upper() == 'Y':
                    continue
                else:
                    raise

    def _process_element_identifier(self,line):
        res = re.search(self._pattern_type,line)
        if res is None:
            element_type = 'Unknown'
        else:
            element_type = res.group()
        self._element_types.append(element_type)
        self._elements.append([])


    def _process_line(self, line, status):
            if status == 'node':
                self._process_node(line)
            elif status == 'element':
                self._process_element(line)

    def _process_node(self,line):
        pattern_digit = self._pattern_digit
        # it seems re.findall do not work with ()
        coordinate = [i.group() for i in re.finditer(pattern_digit,line)][1:]
        assert len(coordinate) == 3, "Number of coordinates is not 3!"
        self.nodes.append([float(i) for i in coordinate])

    def _process_element(self,line):
        pattern_integer = self._pattern_integer
        nodeID = re.findall(pattern_integer,line)[1:]
        self.elements[-1].append([int(i) for i in nodeID])

    def get_nodes(self):
        """Return the nodes of the model."""
        return self._nodes

    def get_elements(self):
        """Return the elements of the model.
        
        
        A list with the node indexes of the mesh returned. The node indexes are
        stored in  numpy arrays and the index of nodes starts from 0.
        
        Returns
        -------
        elements: list
            The nodes of the elements.        
        """
        # index of nodes starts from 0
        return self._elements

    def get_element_types(self):
        """Return the element types of the model.
        
        A list containing the names of the mesh elements is returned.
        
        Returns
        -------
        element_types: list
            The types of the elements.
        """
        # index of nodes starts from 0
        return self._element_types

    @property
    def element_types(self):
        """List of the element types."""
        return self._element_types

    @property
    def elements(self):
        """List of element nodes"""
        return self._elements
    @property
    def nodes(self):
        """List of nodes"""
        return self._nodes
    
    def to_dataset(self, title=''):
        """Convert the representation of nodes and elements to `Dataset`."""
        points = [Point(i) for i in self._nodes]
        ds = DataSet(points, title=title)
        cells = []
        for i, elements in enumerate(self._elements[:2]):
            et = self._element_types[i]
            if et == 'S4R' or et == 'S4':
                cell_type = 9
            elif et == 'B31':
                cell_type = 3
            else:
                cell_type = 0
            cells = [Cell(p, cell_type) for p in elements]
            ds.cells.extend(cells)
        return ds
            
        
        

def read_matrix_coordinate(filename, M=None, N=None):
    """Read sparse matrix from abaqus output.
    
    Each column of the file contains three numbers, which are the index of row,
    the index of column and the element in the corresponding position. The
    indexes begin form 1 and the dimensions of the matrix can be defined by
    extra arguments M and N.
    
    Parameters
    ----------
    filename: string
        Name of the file to read. (usually ends with .mtx for abaqus output)
    M: int
        The number of rows of the sparse matrix.
    N: int
        The number of columns of the sparse matrix.
        
    Returns
    -------
    mat: sparse.coo_matrix
        A sparse matrix object.    
    """

    file = open(filename)
    data = []
    i, j = [], []
    for line in file:
        ix_i, ix_j, data_ij = [eval(i) for i in line.split()]
        data.append(data_ij)
        i.append(ix_i-1)
        j.append(ix_j-1)
    M = max(i)+1 if M is None else M
    N = max(j)+1 if N is None else N
    mat = coo_matrix((data, (i, j)), shape=(M, N))
    return mat


if __name__ == '__main__':
    with open('Job-1.inp') as file:
        ip=AbaqusInputFile(file)
    with open('test-1.vtk', 'w') as file:
        ds = ip.to_dataset()
        file.write(dumpvtk(ds))
        


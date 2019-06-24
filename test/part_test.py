#!/usr/bin/env python3

import os
import sys
os.chdir('..')
sys.path.append(os.curdir)

from fem import *
from scipy.sparse.linalg import eigs

def test_shell():
    with open('test/test-2.inp') as file:
        ip=AbaqusInputFile(file)
    nodes = ip.nodes
    elements = ip.elements[0]
    section = SectionShell(STEEL, 0.005, 'hull')
    hull = PartShell(nodes, elements, section)
    ds = hull.to_dataset()
    with open('generated/part_test_hull.vtk', 'w') as file:
        file.write(dumpvtk(ds))

def test_beam():
    with open('test/test-2.inp') as file:
        ip=AbaqusInputFile(file)
    nodes = ip.nodes
    elements = ip.elements[1]
    section = SectionBeam.circular(STEEL, 0.025)
    shaft = PartBeam(nodes, elements, section)
    ds = shaft.to_dataset()
    with open('generated/part_test_shaft.vtk', 'w') as file:
        file.write(dumpvtk(ds))

if __name__ == '__main__':
    test_shell()
    test_beam()
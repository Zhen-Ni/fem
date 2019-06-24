#!/usr/bin/env python3

import os
import sys
os.chdir('..')
sys.path.append(os.curdir)

from fem import *
from scipy.sparse.linalg import eigs


def test_assembly():
    with open('test/test-2.inp') as file:
        ip=AbaqusInputFile(file)
        
    nodes = ip.nodes
    elements = ip.elements[1]
    section = SectionBeam.circular(STEEL, 0.025)
    shaft = PartBeam(nodes, elements, section)
    elements = ip.elements[0]
    section = SectionShell(STEEL, 0.005, 'hull')
    hull = PartShell(nodes, elements, section)

    ass = Assembly([shaft, hull])

    k=1e9    
    ass.add_spring(k,8,1,4338)
    ass.add_spring(k,9,2,4338)
    ass.add_spring(k,10,2,4338)
    ass.add_spring(k,11,1,4338)
    
    ass.add_spring(k,0,1,4339)
    ass.add_spring(k,3,2,4339)
    ass.add_spring(k,4,2,4339)
    ass.add_spring(k,7,1,4339)

    ass.add_spring(k,1,1,4340)
    ass.add_spring(k,2,2,4340)
    ass.add_spring(k,5,2,4340)
    ass.add_spring(k,6,1,4340)

    ds = ass.to_dataset('all')
    with open('generated/assembly_test.vtk', 'w') as file:
        file.write(dumpvtk(ds))


if __name__ == '__main__':
    test_assembly()

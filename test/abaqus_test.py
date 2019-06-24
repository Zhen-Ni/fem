#!/usr/bin/env python3

import os
import sys
os.chdir('..')
sys.path.append(os.curdir)

from fem import *

def test1():
    with open('test/test-1.inp') as file:
        ip=AbaqusInputFile(file)
    with open('generated/abaqus_test_1.vtk', 'w') as file:
        ds = ip.to_dataset()
        file.write(dumpvtk(ds))

def test2():
    with open('test/test-2.inp') as file:
        ip=AbaqusInputFile(file)
    with open('generated/abaqus_test_2.vtk', 'w') as file:
        ds = ip.to_dataset()
        file.write(dumpvtk(ds))


if __name__ == '__main__':
    test1()
    test2()
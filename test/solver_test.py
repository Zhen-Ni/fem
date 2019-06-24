#!/usr/bin/env python3


#!/usr/bin/env python3

import os
import sys
os.chdir('..')
sys.path.append(os.curdir)

from fem import *
from scipy.sparse.linalg import eigs


def solver_test_modal_1_cylinder():
    with open('test/cylinder.inp') as file:
        ip=AbaqusInputFile(file)
        
    nodes = ip.nodes
    elements = ip.elements[0]
    section = SectionShell(STEEL, 0.01, 'hull')
    shell = PartShell(nodes, elements, section)
    ass = Assembly([shell])
    m = Model(ass)
    
    solver = SolverModal(m, order=50,frequency_shift=10)
    step = solver.solve()
    
    dumppvd(step.to_dataset(), 'solver_test_modal_cylinder', 'generated')
    return step


def slver_test_modal_2_strange_shell():
    with open('test/strange-shell.inp') as file:
        ip=AbaqusInputFile(file)
        
    nodes = ip.nodes
    elements = ip.elements[0]
    section = SectionShell(STEEL, 0.1, 'hull')
    shell = PartShell(nodes, elements, section)
    ass = Assembly([shell])
    m = Model(ass)
    
    solver = SolverModal(m, order=50,frequency_shift=10)
    step = solver.solve()
    
    dumppvd(step.to_dataset(), 'solver_test_modal_strange_shell', 'generated')
    return step


def _get_model():
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

    m = Model(ass)
    return m

def solver_test_modal_3():
    m = _get_model()
    
    solver = SolverModal(m, order=50,frequency_shift=10)
    step = solver.solve()
    
    dumppvd(step.to_dataset(), 'solver_test_modal', 'generated')
    return step

def solver_test_harmonic_1_cylinder():
    with open('test/cylinder.inp') as file:
        ip=AbaqusInputFile(file)
        
    nodes = ip.nodes
    elements = ip.elements[0]
    section = SectionShell(STEEL, 0.01, 'hull')
    shell = PartShell(nodes, elements, section)
    ass = Assembly([shell])
    m = Model(ass)
    m.add_force(1,62,0)
    
    solver = SolverHarmonic(m, np.linspace(1,100,100))
    step = solver.solve()
    
    dumppvd(step.to_dataset(), 'solver_test_harmonic_cylinder', 'generated')
    dumpvtk(step.to_dataset(), 'solver_test_harmonic_cylinder', 'generated')
    return step


if __name__ == '__main__':
    step = solver_test_harmonic_1_cylinder()

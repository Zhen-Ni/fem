#!/usr/bin/env python3

import unittest
import fem

class TestSolver(unittest.TestCase):
    def test_modal_beam1d(self):
        width = 0.1
        height = 0.02
        section = fem.BeamSection.rectangular(fem.STEEL, width, height)
        ds = fem.io.read_inp('test/beam-1d.inp')
        part = fem.BeamPart(ds, section)
        asm = fem.Assembly([part])

        # Encastre boundary at x = 0.
        count = 0
        for idx, pt in enumerate(ds.points):
            if -0.001 < pt.x < 0.001:
                count += 1
                asm.add_spring(1e15, idx, fem.DOF.X)
                asm.add_spring(1e15, idx, fem.DOF.Y)
                asm.add_spring(1e15, idx, fem.DOF.Z)
                asm.add_spring(1e15, idx, fem.DOF.RX)
                asm.add_spring(1e15, idx, fem.DOF.RY)
                asm.add_spring(1e15, idx, fem.DOF.RZ)

        model = fem.Model(asm)
        solver = fem.ModalSolver(model)
        results = solver.solve()

        self.assertAlmostEqual(results[0]['frequency'], 16.688, delta=0.8)
        self.assertAlmostEqual(results[1]['frequency'], 82.820, delta=4)
        self.assertAlmostEqual(results[2]['frequency'], 104.16, delta=5)

        self.assertAlmostEqual(abs(results[0].dof(fem.DOF.Z)[-1]),
                               0.5042, delta=0.01)
        self.assertAlmostEqual(abs(results[1].dof(fem.DOF.Y)[-1]),
                               0.5020, delta=0.01)
        self.assertAlmostEqual(abs(results[2].dof(fem.DOF.Z)[-1]),
                               0.5008, delta=0.01)
        

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

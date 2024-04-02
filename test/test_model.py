#!/usr/bin/env python3

import unittest
import numpy as np

import fem


class TestModel(unittest.TestCase):
    def test_beam1d(self):
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
        # Load at x = 0.
        idx = np.argmin([(p - fem.Point(1, 0.05, 0.01)).norm()
                         for p in ds.points])
        force = 1000
        model.add_force(force, idx, fem.DOF.Z)
        F = model.F
        self.assertTrue(F.sum(), force)
        self.assertEqual(F.tolist().count(0), len(F) - 1)

        # Gravity
        model.add_gravity(-9.8)
        F = model.F
        self.assertEqual(F.tolist().count(0), len(F) / 6 * 5)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

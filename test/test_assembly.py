#!/usr/bin/env python3

import unittest
import fem


class TestAssembly(unittest.TestCase):
    def test_beam1d(self):
        width = 0.1
        height = 0.02
        section = fem.BeamSection.rectangular(fem.STEEL, width, height)
        ds = fem.io.read_inp('test/beam-1d.inp')
        part = fem.BeamPart(ds.points, ds.cells, section)
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
        self.assertEqual(count, 1)
        self.assertTrue(asm.M.nnz > 0)
        self.assertTrue(asm.C.nnz == 0)
        self.assertTrue(asm.K.nnz > 0)

    def test_beam2d(self):
        thickness = 0.02
        section = fem.ShellSection(fem.STEEL, thickness)
        ds = fem.io.read_inp('test/beam-2d.inp')
        part = fem.ShellPart(ds.points, ds.cells, section)
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
        self.assertEqual(count, 6)
        self.assertTrue(asm.M.nnz > 0)
        self.assertTrue(asm.C.nnz == 0)
        self.assertTrue(asm.K.nnz > 0)

    # def test_beam3d(self):
    #     section = fem.SolidSection(fem.STEEL)
    #     ds = fem.io.read_inp('test/beam-3d.inp')
    #     part = fem.SolidPart(ds.points, ds.cells, section)
    #     asm = fem.Assembly([part])

    #     # Encastre boundary at x = 0.
    #     count = 0
    #     for idx, pt in enumerate(ds.points):
    #         if -0.001 < pt.x < 0.001:
    #             count += 1
    #             asm.add_spring(1e15, idx, fem.DOF.X)
    #             asm.add_spring(1e15, idx, fem.DOF.Y)
    #             asm.add_spring(1e15, idx, fem.DOF.Z)
    #             asm.add_spring(1e15, idx, fem.DOF.RX)
    #             asm.add_spring(1e15, idx, fem.DOF.RY)
    #             asm.add_spring(1e15, idx, fem.DOF.RZ)
    #     self.assertEqual(count, 36)
    #     self.assertTrue(asm.M.nnz > 0)
    #     self.assertTrue(asm.C.nnz == 0)
    #     self.assertTrue(asm.K.nnz > 0)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

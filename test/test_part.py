#!/usr/bin/env python3

import unittest
import numpy as np
import fem


class TestPart(unittest.TestCase):
    def test_beampart(self):
        r = 0.1
        section = fem.BeamSection.circular(fem.STEEL, 0.1)
        nodes = fem.Points([fem.Point(0), fem.Point(
            1), fem.Point(2), fem.Point(3), fem.Point(4)])
        elements = fem.Cells.from_array(fem.Line, np.array(
            [0, 1, 1, 2, 2, 3]).reshape(3, 2))
        part = fem.BeamPart(fem.Mesh(nodes, elements), section)
        M = part.M
        K = part.K
        self.assertEqual(M.shape, (30, 30))
        self.assertEqual(K.shape, (30, 30))
        self.assertNotEqual(M.nnz, 0)
        self.assertNotEqual(K.nnz, 0)

    def test_shellpart(self):
        r = 0.1
        section = fem.ShellSection(fem.STEEL, 0.1)
        nodes = fem.Points([fem.Point(0, 0, 0), fem.Point(
            1, 0, 0), fem.Point(1, 1, 0), fem.Point(0, 0, 1)])
        elements = fem.Cells.from_array(fem.Quad, np.array(
            [[0, 1, 2, 2]]))
        part = fem.ShellPart(fem.Mesh(nodes, elements), section)
        M = part.M
        K = part.K
        self.assertEqual(M.shape, (24, 24))
        self.assertEqual(K.shape, (24, 24))
        self.assertNotEqual(M.nnz, 0)
        self.assertNotEqual(K.nnz, 0)


    def test_solidpart(self):
        section = fem.SolidSection(fem.STEEL)
        nodes = fem.Points.from_array([[0, 0, 0],
                                       [0, 0, 1],
                                       [0, 1, 1],
                                       [0, 1, 0],
                                       [1, 0, 0],
                                       [1, 0, 1],
                                       [1, 1, 1],
                                       [1, 1, 0]]
                                      )
        elements = fem.Cells.from_array(fem.Hexahedron, np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7]]))
        part = fem.SolidPart(fem.Mesh(nodes, elements), section)
        M = part.M
        K = part.K
        self.assertEqual(M.shape, (48, 48))
        self.assertEqual(K.shape, (48, 48))
        self.assertNotEqual(M.nnz, 0)
        self.assertNotEqual(K.nnz, 0)


        
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

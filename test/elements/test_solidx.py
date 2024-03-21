#!/usr/bin/env python3

import unittest
import numpy as np
import skfem

import fem
import fem.elements.solidx as solidx


class TestSolidX(unittest.TestCase):
    def test_hexahedron_map(self):
        mesh_ref = skfem.MeshHex1()
        points = fem.Points.from_array(np.array(
            [[0., 0., 0., 1., 0., 1., 1., 1.],
             [0., 0., 1., 0., 1., 0., 1., 1.],
             [0., 1., 0., 0., 1., 1., 0., 1.]]).T)
        cells = fem.Cells.from_array(fem.Hexahedron,
                                     [[1, 5, 3, 0, 4, 7, 6, 2]])
        mesh = solidx.HexahedronMap.to_skfem(fem.Mesh(points, cells))
        self.assertTrue(np.allclose(mesh.doflocs, mesh_ref.doflocs))
        self.assertTrue(np.allclose(mesh.t, mesh_ref.t))
        self.assertEqual(mesh.elem, mesh_ref.elem)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

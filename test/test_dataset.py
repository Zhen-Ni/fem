#!/usr/bin/env python3

import unittest

import numpy as np

import fem


class TestPoints(unittest.TestCase):
    def test_points(self):
        x0, y0, z0 = 0.1, 0.2, 0.3
        x1, y1, z1 = 1.1, 1.2, 1.3
        x2, y2, z2 = 2.1, 2.2, 0.0
        x3, y3, z3 = 0.0, 0.0, 3.3
        p0 = fem.Point(x0, y0, z0)
        p1 = fem.Point(x1, y1, z1)
        p2 = fem.Point(x2, y2)
        p3 = fem.Point(z=z3)
        point_list = [p0, p1, p2, p3]
        points = fem.Points(point_list)
        self.assertEqual(len(points), 4)
        self.assertEqual(list(points.x), [x0, x1, x2, x3])
        self.assertEqual(list(points.y), [y0, y1, y2, y3])
        self.assertEqual(list(points.z), [z0, z1, z2, z3])
        self.assertEqual(points[0], p0)
        self.assertEqual(points[1], p1)
        self.assertEqual(points[2], p2)
        self.assertEqual(points[3], p3)

        points2 = points[:2]
        self.assertEqual(len(points2), 2)
        self.assertEqual(list(points2.x), [x0, x1])
        self.assertEqual(list(points2.y), [y0, y1])
        self.assertEqual(list(points2.z), [z0, z1])
        self.assertEqual(points2[0], p0)
        self.assertEqual(points2[1], p1)

        self.assertEqual(points2, points[[0, 1]])
        self.assertEqual(points2, points[(0, 1)])

        self.assertEqual(repr(points2), "Points object with 2 points")

    def test_points_conversion(self):
        array = np.array([0.1, 0.2, 0.3, 1.1, 1.2, 1.3]).reshape(-1, 3)
        points = fem.Points.from_array(array)
        self.assertEqual(len(points), 2)
        l = points.to_list()
        self.assertEqual(l[0], fem.Point(*array[0]))
        self.assertEqual(l[1], fem.Point(*array[1]))
        self.assertTrue(np.allclose(array, points.to_array()))


class TestCells(unittest.TestCase):

    def test_cells(self):
        l0 = fem.Line(0, 1)
        l1 = fem.Line(1, 2)
        l2 = fem.Line(2, 3)
        cells = fem.Cells([l0, l1, l2])
        self.assertEqual(len(cells), 3)
        self.assertEqual(cells[0], l0)
        self.assertEqual(cells[1], l1)
        self.assertEqual(cells[2], l2)
        self.assertEqual(cells[0].nodes, (0, 1))
        self.assertEqual(cells[1].nodes, (1, 2))
        self.assertEqual(cells[2].nodes, (2, 3))
        self.assertTrue(cells.same_cell_type() is fem.Line)
        
        cells2 = cells[:2]
        self.assertEqual(len(cells2), 2)
        self.assertEqual(cells2[0], l0)
        self.assertEqual(cells2[1], l1)
        self.assertEqual(cells2[0].nodes, (0, 1))
        self.assertEqual(cells2[1].nodes, (1, 2))

        self.assertEqual(cells2, cells[[0, 1]])
        self.assertEqual(cells2, cells[(0, 1)])

        self.assertEqual(repr(cells2), "Cells object with 2 cells")

        self.assertEqual(list(cells), [cells[0], cells[1], cells[2]])

    def test_cells_mixed(self):
        l0 = fem.Vertex(0)
        l1 = fem.Line(0, 1)
        l2 = fem.Quad(0, 1, 2, 3)
        l3 = fem.Hexahedron(0, 1, 2, 3, 4, 5, 6, 7)
        cells = fem.Cells([l0, l1, l2, l3])
        self.assertEqual(len(cells), 4)
        self.assertTrue(cells.same_cell_type() is None)
        l = cells.to_list()
        self.assertEqual(l[0], l0)
        self.assertEqual(l[1], l1)
        self.assertEqual(l[2], l2)
        self.assertEqual(l[3], l3)

    def test_cells_conversion(self):
        arr = np.array([0, 1, 1, 2, 2, 3]).reshape(3, 2)
        cells = fem.Cells.from_array(
            fem.Line, arr)
        self.assertEqual(len(cells), 3)
        self.assertEqual(cells[0].nodes, (0, 1))
        self.assertEqual(cells[1].nodes, (1, 2))
        self.assertEqual(cells[2].nodes, (2, 3))

        self.assertEqual(cells.to_list()[0].nodes, (0, 1))
        self.assertEqual(cells.to_list()[1].nodes, (1, 2))
        self.assertEqual(cells.to_list()[2].nodes, (2, 3))

        self.assertTrue(np.allclose(cells.to_array(), arr))


class TestField(unittest.TestCase):

    def test_field(self):
        data = [1, 2, 3, 4, 5]
        field = fem.ScalarField(np.array(data))
        self.assertEqual(list(field), data)
        self.assertEqual(repr(field), 'ScalarField object with 5 data')
        self.assertEqual(len(field), 5)

        data = [1, 2, 3, 4, 5, 6]
        field = fem.VectorField(np.array(data).reshape(2, 3))
        self.assertEqual(list(field), [(1, 2, 3), (4, 5, 6)])
        self.assertEqual(repr(field), 'VectorField object with 2 data')
        self.assertEqual(len(field), 2)
        self.assertEqual(field.dimension, 3)
        
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

#!/usr/bin/env python3

import unittest
import io

import fem


class TestVTK(unittest.TestCase):
    def test_write_vtk(self):
        points = [fem.Point(0, 0, 0),
                  fem.Point(1, 0, 0),
                  fem.Point(2, 0, 0),
                  fem.Point(0, 1, 0),
                  fem.Point(1, 1, 0),
                  fem.Point(2, 1, 0),
                  fem.Point(1, 2, 0),
                  fem.Point(2, 2, 0)
                  ]
        cells = [fem.Quad(0, 1, 4, 3),
                 fem.Quad(1, 2, 5, 4),
                 fem.Quad(4, 5, 6, 7)]

        temp = fem.FloatScalarField([0, 1, 2, 1, 2, 3, 3, 4])
        disp = fem.FloatArrayField([[0, 0, 0], [0, 0, 1], [0, 0, 0],
                                    [0, 0, 1], [0, 0, 2], [0, 0, 1],
                                    [0, 0, 1], [0, 0, 0]])
        coor = fem.FloatArrayField([[0, 0], [1, 0], [2, 0],
                                    [0, 1], [1, 1], [2, 1],
                                    [1, 2], [2, 2]])
        dataset = fem.Dataset(fem.Mesh(points, cells), 'test')
        dataset.point_data['temprature'] = temp
        dataset.point_data['displacement'] = disp
        dataset.point_data['coordinate'] = coor

        density = fem.FloatScalarField([1, 1.5, 2])
        dataset.cell_data['density'] = density

        buffer = io.StringIO()
        self.assertTrue(fem.io.write_vtk(buffer, dataset))
        buffer.seek(0)
        vtk_content = buffer.read()
        self.assertTrue(vtk_content)

    def test_write_vtu(self):
        points = [fem.Point(0, 0, 0),
                  fem.Point(1, 0, 0),
                  fem.Point(2, 0, 0),
                  fem.Point(0, 1, 0),
                  fem.Point(1, 1, 0),
                  fem.Point(2, 1, 0),
                  fem.Point(1, 2, 0),
                  fem.Point(2, 2, 0)
                  ]
        cells = [fem.Quad(0, 1, 4, 3),
                 fem.Quad(1, 2, 5, 4),
                 fem.Quad(4, 5, 6, 7)]

        temp = fem.FloatScalarField([0, 1, 2, 1, 2, 3, 3, 4])
        disp = fem.FloatArrayField([[0, 0, 0], [0, 0, 1], [0, 0, 0],
                                    [0, 0, 1], [0, 0, 2], [0, 0, 1],
                                    [0, 0, 1], [0, 0, 0]])
        coor = fem.FloatArrayField([[0, 0], [1, 0], [2, 0],
                                    [0, 1], [1, 1], [2, 1],
                                    [1, 2], [2, 2]])
        dataset = fem.Dataset(fem.Mesh(points, cells), 'test')
        dataset.point_data['temprature'] = temp
        dataset.point_data['displacement'] = disp
        dataset.point_data['coordinate'] = coor

        density = fem.FloatScalarField([1, 1.5, 2])
        dataset.cell_data['density'] = density

        buffer = io.StringIO()
        fem.io.write_vtu(buffer, dataset)
        buffer.seek(0)
        vtk_content = buffer.read()
        self.assertTrue(vtk_content)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

#!/usr/bin/env python3

import unittest
import fem


class TestAbaqus(unittest.TestCase):
    def test_read_inp(self):
        ds1 = fem.io.read_inp('./test/beam-1d.inp')
        self.assertTrue(len(ds1.points) > 0)
        self.assertTrue(ds1.cells.same_cell_type(), fem.Line)

        ds2 = fem.io.read_inp('./test/beam-2d.inp')
        self.assertTrue(len(ds2.points) > 0)
        self.assertTrue(ds2.cells.same_cell_type(), fem.Quad)

        ds3 = fem.io.read_inp('./test/beam-3d.inp')
        self.assertTrue(len(ds3.points) > 0)
        self.assertTrue(ds3.cells.same_cell_type(), fem.Hexahedron)

        # The node ids should be smaller than the number of points
        for ds in (ds1, ds2, ds3):
            for cell in ds.cells:
                for n in cell.nodes:
                    self.assertTrue(n < len(ds.points))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

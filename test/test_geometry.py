#!/usr/bin/env python3

import unittest
import fem


class TestGeometry(unittest.TestCase):
    def test_point(self):
        p0 = fem.Point(0., 0., 0.)
        p1 = fem.Point(1., 2., 3.)
        v0 = p1 - p0
        self.assertTrue(v0, fem.Vector(1., 2., 3.))
        self.assertTrue(v0 + p0, p1)
        self.assertTrue(p0 + v0, p1)
        self.assertTrue(p1 - v0, p0)

    def test_vector(self):
        v0 = fem.Vector(1., 2., 3.)
        self.assertEqual(v0.x, 1.)
        self.assertEqual(v0.y, 2.)
        self.assertEqual(v0.z, 3.)

        self.assertEqual(-v0, fem.Vector(-1, -2, -3))
        self.assertEqual(v0 * 2, fem.Vector(2, 4, 6))
        self.assertEqual(2 * v0, v0 * 2)
        self.assertEqual(2 * v0, v0 * 2)
        self.assertEqual(v0 / 0.5, v0 * 2)

        v1 = fem.Vector(3., 4.)
        self.assertEqual(v1.z, 0)
        self.assertEqual(v1.normalize(), v1 / 5)
        self.assertEqual(v1.norm(), 5)

        self.assertEqual(v1 + v0, fem.Vector(4, 6, 3))
        self.assertEqual(v1 + v0, v0 + v1)
        self.assertEqual(v1 - v0, fem.Vector(2, 2, -3))
        self.assertEqual(v1 - v0, -(v0 - v1))

        self.assertEqual(v1 @ v0, 11)
        self.assertEqual(v1 @ v0, v0 @ v1)

        self.assertEqual(fem.E1.cross(fem.E2), fem.E3)
        self.assertEqual(fem.E2.cross(fem.E3), fem.E1)
        self.assertEqual(fem.E3.cross(fem.E1), fem.E2)
        self.assertEqual(fem.E2.cross(fem.E1), -fem.E3)
        self.assertEqual(fem.E3.cross(fem.E2), -fem.E1)
        self.assertEqual(fem.E1.cross(fem.E3), -fem.E2)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

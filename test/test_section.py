#!/usr/bin/env python3

import math
import unittest
import fem


class TestSection(unittest.TestCase):
    def test_solid(self):
        steel = fem.STEEL
        name = 'solid section'
        section = fem.SolidSection(steel, name)
        self.assertEqual(section.material, steel)
        self.assertEqual(repr(section), f'SolidSection: {name}')

    def test_shell(self):
        steel = fem.STEEL
        h = 0.01
        name = 'Raft'
        section = fem.ShellSection(steel, h, name)
        self.assertEqual(section.h, h)
        self.assertEqual(section.name, name)
        self.assertEqual(section.material, steel)
        self.assertEqual(section.kappa, 5. / 6)
        self.assertEqual(repr(section), f'ShellSection: {name}')

    def test_circular_beam(self):
        steel = fem.ALUMINIUM
        r = 0.01
        section = fem.BeamSection.circular(steel, r)
        self.assertAlmostEqual(section.A, math.pi * r ** 2)
        self.assertAlmostEqual(section.Ix, 0.5 * math.pi * r ** 5)
        self.assertEqual(section.name, 'unnamed section')


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

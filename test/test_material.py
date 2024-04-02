#!/usr/bin/env python3

import unittest
import fem


class TestMaterial(unittest.TestCase):
    def test_steel(self):
        material = fem.STEEL
        self.assertEqual(material.E, 210e9)
        self.assertEqual(material.nu, 0.3)
        self.assertEqual(material.rho, 7800)
        self.assertEqual(material.name, 'Steel')
        self.assertEqual(repr(material), 'Material: Steel')

    def test_unnamed(self):
        aluminium = fem.ALUMINIUM
        material = fem.Material(aluminium.E, aluminium.nu, aluminium.rho)
        self.assertEqual(material.name, 'unnamed material')


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

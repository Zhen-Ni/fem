#!/usr/bin/env python3

import unittest
import numpy as np
from scipy.sparse import csc_matrix
import fem
from fem.solver import drop_inactive_dof


class TestRemoveEmptyDofs(unittest.TestCase):
    def test_1(self):
        mat = np.array([
            0, 3, 0, 0, 0,
            22, 0, 0, 0, 17,
            7, 5, 0, 1, 0,
            0, 0, 0, 0, 0,
            0, 0, 14, 0, 8]).reshape(5, 5)
        spmat = csc_matrix(mat)
        (spmat1,), kept_dofs = drop_inactive_dof(spmat)
        self.assertTrue(np.allclose(spmat.toarray(), spmat1.toarray()))
        self.assertEqual(kept_dofs.tolist(), [True] * 5)
        (spmat2,), kept_dofs = drop_inactive_dof(spmat.tocsr())
        self.assertTrue(np.allclose(spmat2.toarray(), spmat1.toarray()))
        self.assertEqual(kept_dofs.tolist(), [True] * 5)

        mat = np.array([
            0, 3, 0, 0, 0,
            22, 0, 0, 0, 17,
            7, 5, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 14, 0, 8]).reshape(5, 5)
        spmat = csc_matrix(mat)
        (spmat1,), kept_dofs = drop_inactive_dof(spmat)
        self.assertTrue(np.allclose(spmat1.toarray(),
                                    mat[[0, 1, 2, 4]].T[[0, 1, 2, 4]].T))
        self.assertEqual(kept_dofs.tolist(), [True, True, True, False, True])
        (spmat2,), kept_dofs = drop_inactive_dof(spmat.tocsr())
        self.assertTrue(np.allclose(spmat2.toarray(), spmat1.toarray()))
        self.assertEqual(kept_dofs.tolist(), [True, True, True, False, True])


class TestSolver(unittest.TestCase):

    def test_modal_beam1d(self):
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
        solver = fem.ModalSolver(model)
        results = solver.solve()

        self.assertAlmostEqual(results[0]['frequency'], 16.688, delta=0.8)
        self.assertAlmostEqual(results[1]['frequency'], 82.820, delta=4)
        self.assertAlmostEqual(results[2]['frequency'], 104.16, delta=5)

        self.assertAlmostEqual(abs(results[0].dof(fem.DOF.Z)[-1]),
                               0.5042, delta=0.01)
        self.assertAlmostEqual(abs(results[1].dof(fem.DOF.Y)[-1]),
                               0.5020, delta=0.01)
        self.assertAlmostEqual(abs(results[2].dof(fem.DOF.Z)[-1]),
                               0.5008, delta=0.01)

    def test_static_beam1d(self):
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

        solver = fem.StaticSolver(model)
        frame = solver.solve()[0]
        self.assertAlmostEqual(frame.translation().norm().max(), 2.381e-2,
                               delta=0.01e-2)

    def test_modal_beam1d_free_free(self):
        width = 0.1
        height = 0.02
        section = fem.BeamSection.rectangular(fem.STEEL, width, height)
        ds = fem.io.read_inp('test/beam-1d.inp')
        part = fem.BeamPart(ds, section)
        asm = fem.Assembly([part])
        model = fem.Model(asm)
        solver = fem.ModalSolver(model)
        results = solver.solve()

        # Should have 6 rigid body modes.
        for i in range(6):
            self.assertAlmostEqual(results[i]['frequency'], 0.0, delta=1e-3)

    def test_static_beam2d(self):
        thickness = 0.02
        steel = fem.Material(210e9, 0.3, 7850)
        section = fem.ShellSection(steel, thickness)
        ds = fem.io.read_inp('test/beam-2d.inp')
        part = fem.ShellPart(ds, section)
        asm = fem.Assembly([part])

        # Encastre boundary at x = 0.
        for idx, pt in enumerate(ds.points):
            if -0.001 < pt.x < 0.001:
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

        solver = fem.StaticSolver(model)
        frame = solver.solve()[0]
        self.assertAlmostEqual(frame.translation().norm().max(), 2.364e-2,
                               delta=0.01e-2)

    def test_modal_beam3d(self):
        steel = fem.Material(210e9, 0.3, 7850)
        section = fem.SolidSection(steel)
        ds = fem.io.read_inp('test/beam-3d.inp')

        part = fem.SolidPart(ds, section)
        asm = fem.Assembly([part])

        # Encastre boundary at x = 0.
        for idx, pt in enumerate(ds.points):
            if -0.001 < pt.x < 0.001:
                asm.add_spring(1e15, idx, fem.DOF.X)
                asm.add_spring(1e15, idx, fem.DOF.Y)
                asm.add_spring(1e15, idx, fem.DOF.Z)
                asm.add_spring(1e15, idx, fem.DOF.RX)
                asm.add_spring(1e15, idx, fem.DOF.RY)
                asm.add_spring(1e15, idx, fem.DOF.RZ)

        model = fem.Model(asm)
        solver = fem.ModalSolver(model)
        results = solver.solve()

        # Natural frequency here is significantly larger due to
        # shear-locking effect. Better result can be obtained by using
        # reduced integration or second-order elements.
        self.assertAlmostEqual(results[0]['frequency'], 19.795, delta=0.8)
        self.assertAlmostEqual(results[1]['frequency'], 83.228, delta=4)
        self.assertAlmostEqual(results[2]['frequency'], 123.76, delta=5)

        self.assertAlmostEqual(results[0].translation().norm().max(),
                               0.5050, delta=0.01)
        self.assertAlmostEqual(results[1].translation().norm().max(),
                               0.5038, delta=0.01)
        self.assertAlmostEqual(results[2].translation().norm().max(),
                               0.5044, delta=0.01)

    def test_static_beam3d(self):
        steel = fem.Material(210e9, 0.3, 7850)
        section = fem.SolidSection(steel)
        ds = fem.io.read_inp('test/beam-3d.inp')
        part = fem.SolidPart(ds, section)
        asm = fem.Assembly([part])

        # Encastre boundary at x = 0.
        for idx, pt in enumerate(ds.points):
            if -0.001 < pt.x < 0.001:
                asm.add_spring(1e15, idx, fem.DOF.X)
                asm.add_spring(1e15, idx, fem.DOF.Y)
                asm.add_spring(1e15, idx, fem.DOF.Z)
                asm.add_spring(1e15, idx, fem.DOF.RX)
                asm.add_spring(1e15, idx, fem.DOF.RY)
                asm.add_spring(1e15, idx, fem.DOF.RZ)
        model = fem.Model(asm)

        # Load at x = 1.
        idx = np.argmin([(p - fem.Point(1, 0.0, 0.0)).norm()
                         for p in ds.points])
        force = 1000
        model.add_force(force, idx, fem.DOF.Z)
        solver = fem.StaticSolver(model)
        results = solver.solve()

        # Deformation is significantly smaller due to shear-locking.
        self.assertAlmostEqual(results[0].translation().norm().max(),
                               1.711e-2, delta=0.1e-2)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

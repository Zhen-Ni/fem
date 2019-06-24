#!/usr/bin/env python3

"""Generate the mass and stiffness matrixes for two node beam elements."""

import numpy as np
from numpy.linalg import norm
from scipy.sparse import coo_matrix, csc_matrix

from .misc import blkdiag


def get_mass_matrix_beam(nodes, elements, section, n1, **kwargs):
    n1 = np.asarray(n1)
    index_row = []
    index_col = []
    data = []

    for element in elements:
        node0, node1 = element
        # process geomentry
        r0, r1 = nodes[node0], nodes[node1]
        e1 = r1 - r0
        l = norm(e1)
        # form the element mass matrix
        Me = get_mass_matrix_beam_sigle_element_lumped_mass(section.rho,
                                                            section.A,
                                                            section.Ix, l)
        # transformation of coordinates
        # 见《MATLAB有限元结构动力学分析与工程应用P124，式(5-107)》
        e1 = e1 / norm(e1)
        e3 = np.cross(e1, n1)
        e3 = e3 / norm(e3)
        e2 = np.cross(e3, e1)
        t = np.array([e1, e2, e3])
        T = blkdiag([t, t, t, t])
        Me = T.T.dot(Me).dot(T)

        # assemble the element mass matrix
        index_bases = node0 * 6, node1 * 6
        for i in range(len(index_bases)):
            index0 = index_bases[i]
            for j in range(len(index_bases)):
                index1 = index_bases[j]
                for ii in range(6):
                    for jj in range(6):
                        value = Me[i * 6 + ii, j * 6 + jj]
                        if value:
                            index_row.append(index0 + ii)
                            index_col.append(index1 + jj)
                            data.append(value)

    # form sparse matrix
    size = len(nodes) * 6
    M = coo_matrix((data, (index_row, index_col)), shape=(size, size))

    return M


def get_stiffness_matrix_beam(nodes, elements, section, n1, **kwargs):
    n1 = np.asarray(n1)
    index_row = []
    index_col = []
    data = []

    for element in elements:
        node0, node1 = element
        # process geomentry
        r0, r1 = nodes[node0], nodes[node1]
        e1 = r1 - r0
        l = norm(e1)
        # form the element mass matrix
        Ke = get_stiffness_matrix_beam_sigle_element(section.E, section.G*section.torsion_coefficient,
                                                     section.kappa, section.A,
                                                     section.Ix, section.Iy,
                                                     section.Iz, l)
        # transformation of coordinates
        # 见《MATLAB有限元结构动力学分析与工程应用P124，式(5-107)》
        e1 = e1 / norm(e1)
        e3 = np.cross(e1, n1)
        e3 = e3 / norm(e3)
        e2 = np.cross(e3, e1)
        t = np.array([e1, e2, e3])
        T = blkdiag([t, t, t, t])
        Ke = T.T.dot(Ke).dot(T)

        # assemble the element mass matrix
        index_bases = node0 * 6, node1 * 6
        for i in range(len(index_bases)):
            index0 = index_bases[i]
            for j in range(len(index_bases)):
                index1 = index_bases[j]
                for ii in range(6):
                    for jj in range(6):
                        value = Ke[i * 6 + ii, j * 6 + jj]
                        if value:
                            index_row.append(index0 + ii)
                            index_col.append(index1 + jj)
                            data.append(value)

    # form sparse matrix
    size = len(nodes) * 6
    K = coo_matrix((data, (index_row, index_col)), shape=(size, size))

    return K


def get_mass_matrix_beam_sigle_element_lumped_mass(rho, A, J, l):
    """《MATLAB有限元结构动力学分析与工程应用P122，式(5-104)》"""
    coeff0 = 0.5 * rho * A * l
    coeff1 = 0.5 * rho * J * l
    Me = np.diag([coeff0, coeff0, coeff0, coeff1, 0, 0] * 2)
    return Me


def get_stiffness_matrix_beam_sigle_element(E, G, k, A, Ix, Iy, Iz, l):
    """《MATLAB有限元结构动力学分析与工程应用P123，式(5-105)》"""
    coeff0 = E * A / l
    coeff1 = k * G * A / l
    coeff2 = G * Ix / l
    coeff3 = G * A * l * k / 4
    coeff4 = E * Iy / l
    coeff5 = E * Iz / l
    coeff6 = k * G * A / 2
    Ke00_diag = np.diag([coeff0, coeff1, coeff1, coeff2,
                         coeff3 + coeff4, coeff3 + coeff5])
    Ke00_offdiag = np.zeros([6, 6])
    Ke00_offdiag[1, 5] = coeff6
    Ke00_offdiag[2, 4] = -coeff6
    Ke01_diag = np.diag([-coeff0, -coeff1, -coeff1,
                         -coeff2, coeff3 - coeff4, coeff3 - coeff5])
    Ke01_offdiag = Ke00_offdiag.copy()
    Ke01_offdiag -= Ke01_offdiag.T
    Ke = np.zeros([12, 12])
    Ke[:6, :6] = Ke00_offdiag
    Ke[:6, 6:] = Ke01_diag + Ke01_offdiag
    Ke[6:, 6:] = -Ke00_offdiag
    Ke += Ke.T
    Ke[:6, :6] += Ke00_diag
    Ke[6:, 6:] += Ke00_diag
    return Ke

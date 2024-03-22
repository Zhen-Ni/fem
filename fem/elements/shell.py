#!/usr/bin/env python3

from .mitc import get_mass_matrix, get_stiffness_matrix


def get_mass_matrix_shell(nodes, elements, section):
    density = section.material.rho
    thickness = section.h
    M = get_mass_matrix(density, thickness, elements, nodes)
    return M.tocoo()


def get_stiffness_matrix_shell(nodes, elements, section):
    E = section.material.E
    nu = section.material.nu
    thickness = section.h
    M = get_stiffness_matrix(E, nu, thickness, elements, nodes)
    return M.tocoo()

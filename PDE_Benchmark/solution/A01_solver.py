"""
Module containing functions to discretize and
solve 1D, steady diffusion problems with
constant boundary conditions
"""

import numpy as np


def discretize_domain(N, L):
    """Set control volume number and cell size"""
    dx = L / N
    return dx


def set_coeff_matrix(kappa, N, Qj):
    """return the coefficient matrix A"""

    A = np.zeros((N, N))
    for j in range(0, N):
        if j == 0:
            A[j, j + 1] = -kappa
        elif j == (N - 1):
            A[j, j - 1] = -kappa
        else:
            A[j, j] = 2 * kappa - Qj
            A[j, j - 1] = -kappa
            A[j, j + 1] = -kappa

    A[0, 0] = 3 * kappa - Qj
    A[N - 1, N - 1] = 3 * kappa - Qj
    # print("A = ", A)

    return A


def set_src_array(kappa, N, Ta, Tb, Qu):
    """return the source term array"""
    B = np.full(N, Qu)
    # B = np.zeros(N)
    B[0] = 2 * kappa * Ta + Qu
    B[N - 1] = 2 * kappa * Tb + Qu
    return B


def solve_phi_array(L, N, area, k, Ta, Tb, qu, qj):
    """ apply the numerical scheme to solve for the
    discrete Temperature array """
    # mesh cell centers
    dx = discretize_domain(N, L)
    kappa = k * area / dx
    Qu = qu * area * dx
    Qj = qj * area * dx
    A = set_coeff_matrix(kappa, N, Qj)
    b = set_src_array(kappa, N, Ta, Tb, Qu)
    xcv = np.linspace(dx / 2.0, L - dx / 2.0, N, endpoint=True)
    phi = np.linalg.solve(A, b)
    return xcv, phi
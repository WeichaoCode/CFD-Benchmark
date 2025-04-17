"""
Module containing functions to discretize and
solve 1D, unsteady convection/diffusion problems with
periodic boundary conditions
"""

import numpy as np


def discretize_domains(N, L, alpha, u, Ts):
    """Set control volume number and cell size"""
    dx = L / N
    dt = alpha * dx / u
    Nt = int(Ts / dt)
    return dx, Nt


def set_initial_condition(xcv, L):
    """set the initial distribution of phi"""
    N = xcv.size
    m = L / 4
    s = L / 20
    phi0 = np.zeros_like(xcv)
    for j in range(N):
        phi0[j] = 1.0 * np.exp(-(xcv[j] - m) ** 2 / s ** 2)
    return phi0


def set_coeff_matrix(N, beta):
    A = np.zeros((N, N))
    for j in range(N):
        A[j, j] = 1 + beta
        if j > 0:
            A[j, j - 1] = -beta / 2
        if j < N - 1:
            A[j, j + 1] = -beta / 2
    A[0, -1] = -beta / 2
    A[-1, 0] = -beta / 2

    return A


def set_src_array(N, phi_c, alpha):
    b = np.zeros_like(phi_c)
    for i in range(N):
        if i == 0:
            b[i] = (1 - alpha) * phi_c[i] + alpha * phi_c[-1]
        else:
            b[i] = (1 - alpha) * phi_c[i] + alpha * phi_c[i - 1]
    return b


def solve_phi_array(L, N, Ts, alpha, u, rho, gamma):
    """ apply the numerical scheme to solve for the
    discrete phi array
    """
    dx, Nt = discretize_domains(N, L, alpha, u, Ts)
    dt = Ts / Nt
    beta = 2 * gamma * dt / (rho * dx * dx)
    xcv = np.linspace(dx / 2.0, L - dx / 2.0, N, endpoint=True)
    phi_0 = set_initial_condition(xcv, L)
    phi_c = np.copy(phi_0)

    A = set_coeff_matrix(N, beta)
    # print(A)
    for n in range(Nt + 1):
        b = set_src_array(N, phi_c, alpha)
        phi_n = np.linalg.solve(A, b)
        phi_c = np.copy(phi_n)
        # print("beta = ",beta)
    # print("alpha = ",alpha)
    # print("Nt = ", Nt)
    # print("dt = ",dt)

    return xcv, phi_0, phi_c
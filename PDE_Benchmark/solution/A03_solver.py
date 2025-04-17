"""
Module containing functions to discretize and
solve 1D, steady convection/diffusion problems with
constant boundary conditions
"""

import numpy as np


def discretize_domains(N, L, alpha, u, Ts):
    """Set control volume number and cell size"""
    dx = L / N
    dt = alpha * dx / u
    Nt = Ts / dt
    return dx, Nt


def set_initial_condition(xcv, L):
    """set the initial distribution of phi"""
    N = xcv.size
    m = L / 4
    s = L / 20
    phi0 = np.zeros_like(xcv)
    for j in range(N):
        phi0[j] = 1.0 * np.exp(-(xcv[j] - m) ** 2 / s ** 2)
        # phi0[j] = 1.0*np.exp(-(xcv[j]-m)**2/(2*s**2))
    return phi0


def set_coeff_matrix(N, alpha):
    A = np.zeros((N, N))
    for j in range(1, N - 1):
        A[j, j - 1] = -alpha / 2
        A[j, j] = 1.0
        A[j, j + 1] = alpha / 2
    A[0, 0] = 1. + alpha / 2
    A[0, 1] = alpha / 2
    A[N - 1, N - 1] = 1. - alpha / 2
    A[N - 1, N - 2] = -alpha / 2
    return A


def set_src_array(N, phi_c, alpha, BCa, BCb):
    b = np.copy(phi_c)
    # b = phi_c
    b[0] += alpha * BCa
    b[N - 1] -= alpha * BCb
    return b


def solve_implicit(N, Nt, alpha, BCa, BCb, phi_c):
    """Solve implicit 1D CDS convective flow"""
    A = set_coeff_matrix(N, alpha)
    for n in range(Nt):
        b = set_src_array(N, phi_c, alpha, BCa, BCb)
        phi_n = np.linalg.solve(A, b)
        phi_c = np.copy(phi_n)

    return phi_c


def solve_explicit(N, Nt, alpha, xsch, BCa, BCb, phi_c):
    phi_n = np.zeros_like(phi_c)
    for n in range(Nt):
        # loop over mesh cells
        for j in range(1, N - 1):
            if xsch == 'CDS':
                phi_n[j] = alpha / 2 * phi_c[j - 1] + phi_c[j] - alpha / 2 * phi_c[j + 1]
            elif xsch == 'UDS':
                phi_n[j] = (1 - alpha) * phi_c[j] + alpha * phi_c[j - 1]
        # apply BCs
        if xsch == 'CDS':
            phi_n[0] = (1. - alpha / 2) * phi_c[0] - alpha / 2 * phi_c[1] + alpha * BCa
            # phi_n[0] = phi_c[j] - alpha*((phi_c[1] + phi_c[0])/2 - BCa)
            phi_n[N - 1] = (1. + alpha / 2) * phi_c[N - 1] + alpha / 2 * phi_c[N - 2] - alpha * BCb
            # phi_n[N-1] = phi_c[j] - alpha*((phi_c[N-1] + phi_c[N-2])/2 - BCb)
        elif xsch == 'UDS':
            phi_n[0] = (1.0 - alpha) * phi_c[0] + alpha * BCa
            phi_n[N - 1] = (1.0 - alpha) * phi_c[N - 1] + alpha * phi_c[N - 2]
        else:
            raise ValueError("xsch may be 'CDS' or 'UDS' ")

        # set solution computed at 'n+1' to be solution at 'n' for next time step
        phi_c = np.copy(phi_n)
        # print("In explicit time loop:",phi_n)
    return phi_c


def solve_phi_array(L, N, Ts, alpha, u, BCa, BCb, xsch, tsch):
    """ apply the numerical scheme to solve for the
    discrete phi array
    """

    dx, Nt = discretize_domains(N, L, alpha, u, Ts)
    xcv = np.linspace(dx / 2.0, L - dx / 2.0, N, endpoint=True)
    phi_0 = set_initial_condition(xcv, L)
    phi_c = np.copy(phi_0)
    Nt = int(Nt)

    if tsch == 'explicit':
        phi_c = solve_explicit(N, Nt, alpha, xsch, BCa, BCb, phi_c)
        # print("explicit end phi",Nt, Ts, Ts/Nt)
    elif tsch == 'implicit':
        phi_c = solve_implicit(N, Nt, alpha, BCa, BCb, phi_c)
    else:
        raise ValueError("tsch may be 'explicit' or 'implicit' ")
    alphac = u * Ts / Nt / dx
    return xcv, phi_0, phi_c, alphac
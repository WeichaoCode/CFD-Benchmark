"""
Module containing functions to discretize and
solve 1D, steady convection/diffusion problems with
constant boundary conditions
"""

import numpy as np


def discretize_domain(N,L):
    """Set control volume number and cell size"""
    dx = L/N
    return dx

def set_coeff_matrix(D, F, N, scheme):
    """return the coefficient matrix A"""

    A = np.zeros((N,N))
    if scheme == 1:
        for j in range(0,N):
          if j == 0:
            A[j,j+1] = -(D - F/2)
          elif j == (N-1):
            A[j,j-1] = -(D + F/2)
          else:
            A[j,j] = 2*D
            A[j,j-1] = -(D + F/2)
            A[j,j+1] = -(D - F/2)
        A[0,0] = 3*D + F/2
        A[N-1,N-1] = 3*D - F/2
    elif scheme == 2:
        for j in range(0,N):
          if j == 0:
            A[j,j+1] = -(D + max(-F, 0))
            # A[j,j+1] = -D
          elif j == (N-1):
            A[j,j-1] = -(D + max(0, F))
            # A[j,j-1] = -(D + F)
          else:
            A[j,j] = 2*D + F
            A[j,j-1] = -(D + max(0, F))
            A[j,j+1] = -(D + max(-F, 0))
            # A[j,j+1] = -D
            # A[j,j-1] = -(D + F)
        A[0,0] = 3*D + F
        A[N-1,N-1] = 3*D + F
        # A[0,1] = -D
        # A[N-1,N-2] = -(D + F)
    else:
        raise ValueError("scheme may be 1 (CDS) or 2 (UDS)")
    # print(A)
    return A

def set_src_array(D,F,N,BCa,BCb, scheme):
    """return the source term array"""
    B = np.zeros(N)
    if scheme == 1:
        B[0] = (2*D + F)*BCa
        B[N-1] = (2*D - F)*BCb
    elif scheme == 2:
        B[0] = (2*D + F)*BCa
        B[N-1] = 2*D*BCb
    else:
        raise ValueError("scheme may be 1 (CDS) or 2 (UDS)")
    return B

def solve_phi_array(L,N,gamma,rho,u,BCa,BCb,scheme):
    """ apply the numerical scheme to solve for the
    discrete phi array
    Last parameter indicates approx scheme for conv term:
        1 - central differencing scheme
        2 - upwind differencing scheme
    """
    # mesh cell centers
    dx = discretize_domain(N,L)
    D = gamma/dx
    F = rho*u
    A = set_coeff_matrix(D, F, N, scheme)
    b = set_src_array(D,F,N,BCa,BCb, scheme)
    # print(b)
    xcv = np.linspace(dx/2.0,L-dx/2.0,N,endpoint=True)
    phi = np.linalg.solve(A,b)
    # print('phi:', phi)
    # print('xcv:', xcv)
    return xcv, phi
#!/usr/bin/env python3
import numpy as np
from scipy.optimize import fsolve

# Parameters
n = 3.0
R0 = 5.0
Nr = 101  # number of grid points
r = np.linspace(0, 1, Nr)
dr = r[1] - r[0]

# Initial guess for f based on the provided expression:
# f0(r) = R0^(2/(n-1)) * (1 - r^2)^2
# For n = 3, exponent 2/(3-1) = 1 so:
f_initial = R0 * (1 - r**2)**2

def residual(f):
    res = np.zeros_like(f)
    # At r=0, using expansion: f''(0) ~ 2*(f[1]-f[0])/(dr^2), and f(0)^n term.
    res[0] = 2.0 * (f[1] - f[0]) / (dr**2) + f[0]**n
    # Interior points i = 1, ..., Nr-2
    for i in range(1, Nr-1):
        # When r[i] is very small, avoid division by zero: but i=0 handled separately.
        # Central differences for f'' and f'
        laplacian = (f[i+1] - 2*f[i] + f[i-1]) / (dr**2)
        # Use central difference for f' and divide by r[i]
        grad_term = (2.0 / r[i]) * ((f[i+1] - f[i-1]) / (2.0 * dr))
        res[i] = laplacian + grad_term + f[i]**n
    # Boundary condition at r=1: f = 0
    res[-1] = f[-1]
    return res

# Solve the nonlinear system using fsolve
f_sol, info, ier, mesg = fsolve(residual, f_initial, full_output=True)

if ier != 1:
    raise RuntimeError("Nonlinear solver did not converge: " + mesg)

# Save the final solution as a 1D NumPy array in f.npy
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/f_sol_Lane_Emden_Equation.npy', f_sol)

if __name__ == '__main__':
    pass
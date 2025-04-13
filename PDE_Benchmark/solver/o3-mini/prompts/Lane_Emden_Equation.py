#!/usr/bin/env python3
import numpy as np
from scipy.integrate import solve_bvp

# Problem parameters
n = 3.0

# Define the ODE system for the Lane-Emden equation:
# Let y[0] = f, y[1] = f'.
# Then, f' = y[1] and f'' = - (2/r)*y[1] - f^n.
# At r=0, by symmetry, f'(0)=0 and the singular term is replaced by - f(0)^n.
def ode_system(r, y):
    dydr = np.zeros_like(y)
    dydr[0] = y[1]
    # Replace r by a small number near zero to avoid division by zero
    r_safe = np.where(np.abs(r) < 1e-8, 1e-8, r)
    dydr[1] = -2.0/r_safe * y[1] - y[0]**n
    # For r very near 0, enforce the limiting behavior: f''(0) = -f(0)^n.
    near_zero = np.abs(r) < 1e-8
    if np.any(near_zero):
        dydr[1, near_zero] = - y[0, near_zero]**n
    return dydr

# Define the boundary conditions:
# At r = 0, regularity: f'(0) = 0.
# At r = 1, Dirichlet condition: f(1) = 0.
def bc(ya, yb):
    return np.array([ya[1], yb[0]])

# Construct a radial mesh on the domain [0, 1]
N = 500
r = np.linspace(0, 1, N)

# Define an initial guess that satisfies the boundary conditions.
# A simple polynomial: f(r) = 1 - r^2, so that f(1)=0 and f'(0)=0.
def initial_guess(r):
    f_guess = 1 - r**2
    df_dr_guess = -2*r
    return np.vstack((f_guess, df_dr_guess))

y_guess = initial_guess(r)

# Solve the boundary value problem using solve_bvp with generous max_nodes.
solution = solve_bvp(ode_system, bc, r, y_guess, tol=1e-5, max_nodes=10000)

# If the solver did not converge, try refining the mesh.
if not solution.success:
    N_refined = 1000
    r_refined = np.linspace(0, 1, N_refined)
    y_guess_refined = initial_guess(r_refined)
    solution = solve_bvp(ode_system, bc, r_refined, y_guess_refined, tol=1e-5, max_nodes=20000)
    if not solution.success:
        raise RuntimeError("BVP solver did not converge")

# Use the mesh from the final successful solve.
r_final = solution.x
f = solution.sol(r_final)[0]

# Save the final 1D solution array in "f.npy"
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/f_Lane_Emden_Equation.npy', f)
#!/usr/bin/env python3
import numpy as np

# Parameters
alpha = 0.01            # thermal diffusivity
Q0 = 200.0              # source term amplitude (°C/s)
sigma = 0.1             # spread of source term
t_final = 3.0           # final time
nx = 101                # number of grid points in x
ny = 101                # number of grid points in y

# Spatial domain
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

# Time stepping: stability criterion for explicit scheme: dt <= min(dx^2,dy^2)/(4*alpha)
dt = min(dx, dy)**2 / (4 * alpha) * 0.5  # use some safety factor
nt = int(t_final / dt)
dt = t_final / nt  # adjust dt to exactly reach t_final

# Meshgrid for evaluation of initial and source term
X, Y = np.meshgrid(x, y, indexing='ij')

# Initial condition
T = 1.0 + 200.0 * np.exp(-((X**2 + Y**2) / (2 * sigma**2)))

# Source term (independent of time in this problem formulation)
q = Q0 * np.exp(-((X**2 + Y**2) / (2 * sigma**2)))

# Boundary condition value
T_boundary = 1.0

# Time integration (explicit Euler with central differences)
for step in range(nt):
    T_new = T.copy()
    # Update interior nodes
    T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + dt * (
        alpha * (
            (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2 +
            (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2
        ) + q[1:-1, 1:-1]
    )
    # Enforce boundary conditions: T = 1 on all boundaries
    T_new[0, :] = T_boundary
    T_new[-1, :] = T_boundary
    T_new[:, 0] = T_boundary
    T_new[:, -1] = T_boundary
    T = T_new

# Save final solution in a .npy file: as a 2D array since the problem is 2D.
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/T_2D_Unsteady_Heat_Equation.npy', T)
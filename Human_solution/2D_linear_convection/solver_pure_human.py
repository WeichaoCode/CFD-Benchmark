"""
=========================================================================
2D Linear Convection Solver with Method of Manufactured Solutions (MMS)
=========================================================================

Author: Weichao Li
Email: liw19111201@gmail.com
Date: 2025-02-21
Version: 1.0

Description:
    This script implements a numerical solver for the 2D Linear Convection
    equation using the Finite Difference Method (FDM). The Method of Manufactured
    Solutions (MMS) is used to provide an exact solution for validation.

    The PDE being solved:
        ∂u/∂t + c_x * ∂u/∂x + c_y * ∂u/∂y = f(x, y, t)

    where the manufactured solution is:
        u(x, y, t) = exp(-t) * sin(pi * x * y)

    The source term f(x, y, t) is derived to ensure the solution satisfies
    the PDE exactly.

Features:
    - Supports multiple numerical schemes: FTCS, Upwind, Lax-Wendroff
    - Computes source term dynamically using MMS
    - Applies exact boundary and initial conditions from MMS
    - Compares numerical solution with exact MMS solution
    - Visualizes results using 2D and 3D plots

Usage:
    To run this script, simply execute:
        python solve_2d_convection.py

    Example:
        solver = Solve2DLinearConvection(nx=50, ny=50, nt=200, T=2.0,
                                         Lx=1.0, Ly=1.0, cx=1.0, cy=1.0,
                                         method="Lax-Wendroff")
        solver.solve()
        solver.plot_solution()

Dependencies:
    - numpy
    - matplotlib

License:
    MIT License. You are free to modify and distribute this script with proper
    attribution to the original author.
"""

import numpy as np
import matplotlib.pyplot as plt


# Define parameters
nx, ny = 50, 50  # Grid points
nt = 200  # Time steps
T = 2.0  # Final time
Lx, Ly = 1.0, 1.0  # Domain size
cx, cy = 1.0, 1.0  # Convection speeds

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = T / (nt - 1)

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
t = np.linspace(0, T, nt)


# source term
def f(x_f, y_f, t_f):
    return (-np.exp(-t_f) * np.sin(np.pi * x_f * y_f) + np.pi * cx * y_f * np.exp(-t_f) * np.cos(np.pi * x_f * y_f)
            + cy * np.pi * x_f * np.exp(-t_f) * np.cos(np.pi * x_f * y_f))


# set the initial conditions
u = np.zeros((nx, ny, nt))

for i in range(nx):
    for j in range(ny):
        u[i, j, 0] = np.exp(-t[0]) * np.sin(np.pi * x[i] * y[j])

# set the boundary condition x = 0
for j in range(ny):
    for n in range(nt):
        u[0, j, n] = np.exp(-t[n]) * np.sin(np.pi * x[0] * y[j])

# set the boundary condition y = 0
for i in range(nx):
    for n in range(nt):
        u[i, 0, n] = np.exp(-t[n]) * np.sin(np.pi * x[i] * y[0])

for n in range(nt - 1):
    for i in range(1, nx):
        for j in range(1, ny):
            u[i, j, n + 1] = (u[i, j, n] - cx * dt / dx * (u[i, j, n] - u[i - 1, j, n])
                              - cy * dt / dy * (u[i, j, n] - u[i, j - 1, n]) + dt * f(x[i], y[j], t[n]))

# compute the exact solution
u_exact = np.zeros((nx, ny, nt))
for n in range(nt):
    for i in range(nx):
        for j in range(ny):
            u_exact[i, j, n] = np.exp(-t[n]) * np.sin(np.pi * x[i] * y[j])

# plot the results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Numerical solution
ax1 = axes[0]
c1 = ax1.contourf(x, y, u[..., -1], cmap="viridis")
plt.colorbar(c1, ax=ax1)
ax1.set_title("Numerical Solution")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# MMS (exact) solution
ax2 = axes[1]
c2 = ax2.contourf(x, y, u_exact[..., -1], cmap="viridis")
plt.colorbar(c2, ax=ax2)
ax2.set_title("MMS (Exact) Solution")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

# Absolute Error Plot
error = u[..., -1] - u_exact[..., -1]
ax3 = axes[2]
c3 = ax3.contourf(x, y, error, cmap="inferno")  # Using 'inferno' to highlight errors
plt.colorbar(c3, ax=ax3)
ax3.set_title("Absolute Error |u - MMS|")
ax3.set_xlabel("x")
ax3.set_ylabel("y")

# Display overall title
plt.suptitle(f"Comparison at t = {T}", fontsize=14)
plt.tight_layout()
plt.show()
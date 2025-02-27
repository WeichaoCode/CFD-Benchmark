"""
2D Diffusion Equation Solver using MMS (Stable)
---------------------------------------------------------------
- MMS Solution: u(x, y, t) = exp(-t) * sin(pi * x) * sin(pi * y)
- Finite Difference Method (Explicit Scheme)
- CFL condition enforced for stability
- Dirichlet boundary conditions from MMS
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/02/26
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- PARAMETERS ----------------
nx, ny = 50, 50  # Grid points
nt = 1000  # Time steps
T = 2.0  # Final time
Lx, Ly = 1.0, 1.0  # Domain size
nu = 0.05  # Diffusion coefficient

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = T / (nt - 1)

# ---------------- ENSURE CFL CONDITION ----------------
dt_max = min(dx ** 2, dy ** 2) / (4 * nu)
if dt > dt_max:
    print(f"Reducing dt from {dt:.6f} to {dt_max:.6f} for stability.")
    dt = dt_max

# Define spatial and temporal grids
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
t = np.linspace(0, T, nt)

X, Y = np.meshgrid(x, y, indexing="ij")  # 2D grid


# ---------------- SOURCE TERM FROM MMS ----------------
def f_source(x_f, y_f, t_f):
    return (2 * nu * np.pi ** 2 - 1) * np.exp(-t_f) * np.sin(np.pi * x_f) * np.sin(np.pi * y_f)


# ---------------- INITIAL AND BOUNDARY CONDITIONS FROM MMS ----------------
u = np.zeros((nx, ny, nt))  # Store full time history
u[:, :, 0] = np.exp(-t[0]) * np.sin(np.pi * X) * np.sin(np.pi * Y)  # Initial condition

# ---------------- SOLVE USING FINITE DIFFERENCE ----------------
for n in range(nt - 1):
    f_t = f_source(X, Y, t[n])  # Compute source term at current time step

    # Apply finite difference scheme (Explicit Euler)
    u_new = np.copy(u[:, :, n])

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u_new[i, j] = (u[i, j, n] +
                           nu * dt / dx ** 2 * (u[i + 1, j, n] - 2 * u[i, j, n] + u[i - 1, j, n]) +
                           nu * dt / dy ** 2 * (u[i, j + 1, n] - 2 * u[i, j, n] + u[i, j - 1, n]) +
                           dt * f_t[i, j])

    # Apply MMS boundary conditions at next time step
    u_new[0, :] = np.exp(-t[n + 1]) * np.sin(np.pi * x[0]) * np.sin(np.pi * y)  # x = 0
    u_new[-1, :] = np.exp(-t[n + 1]) * np.sin(np.pi * x[-1]) * np.sin(np.pi * y)  # x = Lx
    u_new[:, 0] = np.exp(-t[n + 1]) * np.sin(np.pi * x) * np.sin(np.pi * y[0])  # y = 0
    u_new[:, -1] = np.exp(-t[n + 1]) * np.sin(np.pi * x) * np.sin(np.pi * y[-1])  # y = Ly

    # Store updated values
    u[:, :, n + 1] = u_new

# ---------------- COMPUTE EXACT SOLUTION FOR COMPARISON ----------------
u_exact = np.zeros((nx, ny, nt))
for n in range(nt):
    u_exact[:, :, n] = np.exp(-t[n]) * np.sin(np.pi * X) * np.sin(np.pi * Y)

# ---------------- ERROR ANALYSIS ----------------
error = np.abs(u - u_exact)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Numerical solution at final time step
ax1 = axes[0]
c1 = ax1.contourf(X, Y, u[:, :, -1], cmap="viridis")
plt.colorbar(c1, ax=ax1)
ax1.set_title("Numerical Solution at t = T")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# MMS (exact) solution at final time step
ax2 = axes[1]
c2 = ax2.contourf(X, Y, u_exact[:, :, -1], cmap="viridis")
plt.colorbar(c2, ax=ax2)
ax2.set_title("MMS (Exact) Solution at t = T")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

# Absolute Error at final time step
ax3 = axes[2]
c3 = ax3.contourf(X, Y, error[:, :, -1], cmap="inferno")  # Using 'inferno' to highlight errors
plt.colorbar(c3, ax=ax3)
ax3.set_title("Absolute Error |u - MMS| at t = T")
ax3.set_xlabel("x")
ax3.set_ylabel("y")

plt.suptitle(f"Comparison at t = {T}", fontsize=14)
plt.tight_layout()
plt.show()

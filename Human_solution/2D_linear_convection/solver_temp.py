"""
==================================================================================
2D Linear Convection Solver with Method of Manufactured Solutions (MMS)
==================================================================================
Author: [Your Name]
Date: [Today's Date]
License: MIT
Description:
    This script solves the 2D linear convection equation using finite difference
    methods (FTCS, Upwind, Lax-Wendroff) with the Method of Manufactured Solutions (MMS).

    The manufactured solution is:
        u(x, y, t) = exp(-t) * sin(pi * x * y)

    The governing equation:
        ∂u/∂t + c_x * ∂u/∂x + c_y * ∂u/∂y = f(x, y, t)

    where f(x, y, t) is computed from the MMS solution.

    This implementation is **vectorized** to avoid explicit loops, making it
    efficient and suitable for high-resolution simulations.

Acknowledgment:
    This code was developed with the assistance of an AI language model (LLM).
==================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Simulation Parameters
# =========================
nx, ny = 50, 50  # Grid points
nt = 200  # Time steps
T = 2.0  # Final simulation time
Lx, Ly = 1.0, 1.0  # Domain size
cx, cy = 1.0, 1.0  # Convection speeds
method = "Upwind"  # Choose from "FTCS", "Upwind", "Lax-Wendroff"

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = T / (nt - 1)

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
t = np.linspace(0, T, nt)

X, Y = np.meshgrid(x, y, indexing="ij")  # Shape: (nx, ny)


# =========================
# Source Term Computation (MMS)
# =========================
def compute_source_term(X, Y, t):
    """Computes the manufactured source term f(x, y, t)."""
    return (-np.exp(-t) * np.sin(np.pi * X * Y) +
            np.pi * cx * Y * np.exp(-t) * np.cos(np.pi * X * Y) +
            np.pi * cy * X * np.exp(-t) * np.cos(np.pi * X * Y))


# Compute source term for all (x, y, t)
f = compute_source_term(X[:, :, None], Y[:, :, None], t[None, None, :])  # Shape: (nx, ny, nt)

# =========================
# Initial and Boundary Conditions
# =========================
u = np.exp(-t[0]) * np.sin(np.pi * X * Y)  # Initial condition at t=0

# Apply boundary conditions for all time steps using broadcasting
u_bc_x = np.exp(-t[:, None]) * np.sin(np.pi * 0 * y)  # x = 0
u_bc_x_L = np.exp(-t[:, None]) * np.sin(np.pi * Lx * y)  # x = Lx
u_bc_y = np.exp(-t[:, None]) * np.sin(np.pi * x * 0)  # y = 0
u_bc_y_L = np.exp(-t[:, None]) * np.sin(np.pi * x * Ly)  # y = Ly

# =========================
# Solve PDE using FDM (Vectorized)
# =========================
for n in range(1, nt - 1):
    # Apply boundary conditions
    u[0, :] = u_bc_x[n]  # x = 0
    u[-1, :] = u_bc_x_L[n]  # x = Lx
    u[:, 0] = u_bc_y[n]  # y = 0
    u[:, -1] = u_bc_y_L[n]  # y = Ly

    if method == "FTCS":
        u[1:-1, 1:-1] = (
                u[1:-1, 1:-1]
                - 0.5 * cx * dt / dx * (u[2:, 1:-1] - u[:-2, 1:-1])
                - 0.5 * cy * dt / dy * (u[1:-1, 2:] - u[1:-1, :-2])
                + dt * f[1:-1, 1:-1, n]
        )

    elif method == "Upwind":
        u[1:-1, 1:-1] = (
                u[1:-1, 1:-1]
                - cx * dt / dx * (u[1:-1, 1:-1] - u[:-2, 1:-1])
                - cy * dt / dy * (u[1:-1, 1:-1] - u[1:-1, :-2])
                + dt * f[1:-1, 1:-1, n]
        )

    elif method == "Lax-Wendroff":
        u[1:-1, 1:-1] = (
                u[1:-1, 1:-1]
                - 0.5 * cx * dt / dx * (u[2:, 1:-1] - u[:-2, 1:-1])
                - 0.5 * cy * dt / dy * (u[1:-1, 2:] - u[1:-1, :-2])
                + 0.5 * (cx * dt / dx) ** 2 * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1])
                + 0.5 * (cy * dt / dy) ** 2 * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])
                + dt * f[1:-1, 1:-1, n]
        )

# =========================
# Compute MMS Exact Solution
# =========================
u_exact = np.exp(-t[None, None, :]) * np.sin(np.pi * X[:, :, None] * Y[:, :, None])  # Shape: (nx, ny, nt)

# =========================
# Plot Results (Final Time)
# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Numerical Solution
c1 = axes[0].contourf(x, y, u, cmap="viridis")
plt.colorbar(c1, ax=axes[0])
axes[0].set_title("Numerical Solution")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# MMS (Exact) Solution
c2 = axes[1].contourf(x, y, u_exact[..., -1], cmap="viridis")
plt.colorbar(c2, ax=axes[1])
axes[1].set_title("MMS (Exact) Solution")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

# Absolute Error Plot
error = np.abs(u - u_exact[..., -1])
c3 = axes[2].contourf(x, y, error, cmap="inferno")
plt.colorbar(c3, ax=axes[2])
axes[2].set_title("Absolute Error |u - MMS|")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")

plt.suptitle(f"Comparison at t = {T} (Method: {method})", fontsize=14)
plt.tight_layout()
plt.show()

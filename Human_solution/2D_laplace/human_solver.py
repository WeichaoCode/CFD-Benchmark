"""
2D Laplace Equation Solver using MMS (Stable)
---------------------------------------------------------------
- MMS Solution: u(x, y, t) = sin(pi * x) * sin(pi * y)
- Dirichlet boundary conditions from MMS
- This is static problem
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/02/27
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------- PARAMETERS ----------------
nx, ny = 50, 50  # Grid points
Lx, Ly = 1.0, 1.0  # Domain size

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
# Define spatial and temporal grids
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing="ij")  # 2D grid


# ---------------- SOURCE TERM FROM MMS ----------------
def f_u(x_f, y_f):
    return -2 * np.pi ** 2 * np.sin(np.pi * x_f) * np.sin(np.pi * y_f)


# ---------------- INITIALIZATION ----------------
u = np.zeros((nx, ny))  # Solution array
f = f_u(X, Y)  # Source term
# Apply Dirichlet boundary conditions from MMS
u[0, :] = np.sin(np.pi * x[0]) * np.sin(np.pi * y)  # Left boundary
u[-1, :] = np.sin(np.pi * x[-1]) * np.sin(np.pi * y)  # Right boundary
u[:, 0] = np.sin(np.pi * x) * np.sin(np.pi * y[0])  # Bottom boundary
u[:, -1] = np.sin(np.pi * x) * np.sin(np.pi * y[-1])  # Top boundary

# ---------------- SOLVE USING ITERATIVE FINITE DIFFERENCE ----------------
max_iter = 10000  # Maximum iterations
tol = 1e-6  # Convergence tolerance

for iteration in range(max_iter):
    u_new = np.copy(u)

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u_new[i, j] = (- dx ** 2 * dy ** 2 * f[i, j] +
                           dy ** 2 * (u[i + 1, j] + u[i - 1, j]) +
                           dx ** 2 * (u[i, j + 1] + u[i, j - 1])) / (2 * (dx ** 2 + dy ** 2))
    # Check for convergence
    error = np.linalg.norm(u_new - u, ord=np.inf)
    if error < tol:
        print(f"Converged in {iteration} iterations.")
        break

    u = u_new
# ---------------- COMPUTE EXACT SOLUTION FOR COMPARISON ----------------
u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

# ---------------- ERROR ANALYSIS ----------------
error = np.abs(u - u_exact)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Numerical solution at final time step
ax1 = axes[0]
c1 = ax1.contourf(X, Y, u, cmap="viridis")
plt.colorbar(c1, ax=ax1)
ax1.set_title("Numerical Solution at t = T")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# MMS (exact) solution at final time step
ax2 = axes[1]
c2 = ax2.contourf(X, Y, u_exact, cmap="viridis")
plt.colorbar(c2, ax=ax2)
ax2.set_title("MMS (Exact) Solution at t = T")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

# Absolute Error at final time step
ax3 = axes[2]
c3 = ax3.contourf(X, Y, error, cmap="inferno")  # Using 'inferno' to highlight errors
plt.colorbar(c3, ax=ax3)
ax3.set_title("Absolute Error |u - MMS| at t = T")
ax3.set_xlabel("x")
ax3.set_ylabel("y")

plt.suptitle(f"Comparison at iter = {iteration}", fontsize=14)
plt.tight_layout()
plt.show()

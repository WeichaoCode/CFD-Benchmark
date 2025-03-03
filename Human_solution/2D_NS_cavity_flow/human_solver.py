"""
2D Cavity Flow with Navier Stokes Equation Solver using MMS (Stable)
---------------------------------------------------------------
- MMS Solution: u(x, y, t) = exp(-t) * sin(pi * x) * cos(pi * y)
                v(x, y, t) = - exp(-t) * cos(pi * x) * sin(pi * y)
                p(x, y, t) = cos(pi * x) * cos(pi * y)
- Dirichlet boundary conditions from MMS
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/02/28
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------- PARAMETERS ----------------
nx, ny = 50, 50  # Grid points
nt = 20  # Time steps
T = 0.2  # Final time
Lx, Ly = 1.0, 1.0  # Domain size
nu, rho = 0.1, 1  # Diffusion coefficient
max_iter = 1000

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
def f_u(x_f, y_f, t_f):
    return ((np.pi * rho * np.cos(np.pi * x_f) - np.pi * np.exp(t_f) * np.cos(np.pi * y_f) +
             rho * (2 * nu * np.pi ** 2 - 1) * np.exp(t_f) * np.cos(np.pi * y_f))
            * np.exp(-2 * t_f) * np.sin(np.pi * x_f) / rho)


def f_v(x_f, y_f, t_f):
    return ((np.pi * rho * np.cos(np.pi * y_f) - np.pi * np.exp(t_f) * np.cos(np.pi * x_f) +
             rho * (-2 * nu * np.pi ** 2 + 1) * np.exp(t_f) * np.cos(np.pi * x_f))
            * np.exp(-2 * t_f) * np.sin(np.pi * y_f) / rho)


def f_p(x_f, y_f, t_f):
    return np.pi ** 2 * (rho * (np.cos(2 * np.pi * x_f) + np.cos(2 * np.pi * y_f)) -
                         2 * np.exp(2 * t_f) * np.cos(np.pi * x_f) * np.cos(np.pi * y_f)) * np.exp(-2 * t_f)


# ---------------- INITIAL AND BOUNDARY CONDITIONS FROM MMS ----------------
u = np.zeros((nx, ny, nt))
v = np.zeros((nx, ny, nt))
p = np.zeros((nx, ny))
u[:, :, 0] = np.exp(-t[0]) * np.sin(np.pi * X) * np.cos(np.pi * Y)
v[:, :, 0] = - np.exp(-t[0]) * np.cos(np.pi * X) * np.sin(np.pi * Y)


# ---------------- SOLVE USING FINITE DIFFERENCE ----------------
def pressure_poisson(p_f, dx_f, dy_f, dt_f, u_f, v_f, t_f):
    dx2, dy2 = dx_f ** 2, dy_f ** 2
    coeff = 1 / (2 * (dx2 + dy2))
    # Apply Dirichlet boundary conditions from MMS
    p_f[0, :] = np.cos(np.pi * x[0]) * np.cos(np.pi * y)  # Left boundary
    p_f[-1, :] = np.cos(np.pi * x[-1]) * np.cos(np.pi * y)  # Right boundary
    p_f[:, 0] = np.cos(np.pi * x) * np.cos(np.pi * y[0])  # Bottom boundary
    p_f[:, -1] = np.cos(np.pi * x) * np.cos(np.pi * y[-1])  # Top boundary
    fp_t = f_p(X, Y, t[n])
    for iteration in range(max_iter):
        p_new = np.copy(p_f)
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                divergence_term = (1 / dt_f) * ((u_f[i + 1, j, t_f] - u_f[i - 1, j, t_f]) / (2 * dx_f) +
                                                (v_f[i, j + 1, t_f] - v_f[i, j - 1, t_f]) / (2 * dy_f))

                nonlinear_term = ((u_f[i + 1, j, t_f] - u_f[i - 1, j, t_f]) / (2 * dx_f)) ** 2 + \
                                 2 * ((u_f[i, j + 1, t_f] - u_f[i, j - 1, t_f]) / (2 * dy_f)) * (
                                         (v_f[i + 1, j, t_f] - v_f[i - 1, j, t_f]) / (2 * dx_f)) + \
                                 ((v_f[i, j + 1, t_f] - v_f[i, j - 1, t_f]) / (2 * dy_f)) ** 2

                source_term = rho * (divergence_term - nonlinear_term) - fp_t[i, j]

                p_new[i, j] = coeff * ((p_f[i + 1, j] + p_f[i - 1, j]) * dy2 +
                                       (p_f[i, j + 1] + p_f[i, j - 1]) * dx2 -
                                       rho * dx2 * dy2 * source_term)
        p_f = p_new
    return p_f


for n in range(nt - 1):
    fu_t = f_u(X, Y, t[n])
    fv_t = f_v(X, Y, t[n])
    u_new = np.copy(u[:, :, n])
    v_new = np.copy(v[:, :, n])
    p = pressure_poisson(p, dx, dy, dt, u, v, n)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u_new[i, j] = (u[i, j, n] - u[i, j, n] * dt / dx * (u[i, j, n] - u[i - 1, j, n])
                           - v[i, j, n] * dt / dy * (u[i, j, n] - u[i, j - 1, n])
                           - dt / (2 * rho * dx) * (p[i + 1, j] - p[i - 1, j])
                           + nu * (dt / dx ** 2 * (u[i + 1, j, n] - 2 * u[i, j, n] + u[i - 1, j, n])
                                   + dt / dy ** 2 * (u[i, j + 1, n] - 2 * u[i, j, n] + u[i, j - 1, n]))
                           + dt * fu_t[i, j])

            v_new[i, j] = (v[i, j, n] - u[i, j, n] * dt / dx * (v[i, j, n] - v[i - 1, j, n])
                           - v[i, j, n] * dt / dy * (v[i, j, n] - v[i, j - 1, n])
                           - dt / (2 * rho * dy) * (p[i, j + 1] - p[i, j - 1])
                           + nu * (dt / dx ** 2 * (v[i + 1, j, n] - 2 * v[i, j, n] + v[i - 1, j, n])
                                   + dt / dy ** 2 * (v[i, j + 1, n] - 2 * v[i, j, n] + v[i, j - 1, n]))
                           + dt * fv_t[i, j])
    # Apply MMS boundary conditions
    u_new[0, :] = np.exp(-t[n + 1]) * np.sin(np.pi * x[0]) * np.cos(np.pi * y)  # Left boundary
    u_new[-1, :] = np.exp(-t[n + 1]) * np.sin(np.pi * x[-1]) * np.cos(np.pi * y)  # Right boundary
    u_new[:, 0] = np.exp(-t[n + 1]) * np.sin(np.pi * x) * np.cos(np.pi * y[0])  # Bottom boundary
    u_new[:, -1] = np.exp(-t[n + 1]) * np.sin(np.pi * x) * np.cos(np.pi * y[-1])  # Top boundary

    v_new[0, :] = - np.exp(-t[n + 1]) * np.cos(np.pi * x[0]) * np.sin(np.pi * y)
    v_new[-1, :] = - np.exp(-t[n + 1]) * np.cos(np.pi * x[-1]) * np.sin(np.pi * y)
    v_new[:, 0] = - np.exp(-t[n + 1]) * np.cos(np.pi * x) * np.sin(np.pi * y[0])
    v_new[:, -1] = - np.exp(-t[n + 1]) * np.cos(np.pi * x) * np.sin(np.pi * y[-1])

    # Store updated values
    u[:, :, n + 1] = u_new
    v[:, :, n + 1] = v_new

# ---------------- COMPUTE EXACT SOLUTION FOR COMPARISON ----------------
u_exact = np.zeros((nx, ny, nt))
v_exact = np.zeros((nx, ny, nt))
for n in range(nt):
    u_exact[:, :, n] = np.exp(-t[n]) * np.sin(np.pi * X) * np.cos(np.pi * Y)
    v_exact[:, :, n] = - np.exp(-t[n]) * np.cos(np.pi * X) * np.sin(np.pi * Y)
p_exact = np.cos(np.pi * X) * np.cos(np.pi * Y)

# ---------------- ERROR ANALYSIS ----------------
error_u = np.abs(u - u_exact)
error_v = np.abs(v - v_exact)
error_p = np.abs(p - p_exact)

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# Numerical solution at final time step
ax1 = axes[0, 0]
c1 = ax1.contourf(X, Y, u[:, :, -1], cmap="viridis")
plt.colorbar(c1, ax=ax1)
ax1.set_title("Numerical Solution at t = T")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# MMS (exact) solution at final time step
ax2 = axes[0, 1]
c2 = ax2.contourf(X, Y, u_exact[:, :, -1], cmap="viridis")
plt.colorbar(c2, ax=ax2)
ax2.set_title("MMS (Exact) Solution at t = T")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

# Absolute Error at final time step
ax3 = axes[0, 2]
c3 = ax3.contourf(X, Y, error_u[:, :, -1], cmap="inferno")  # Using 'inferno' to highlight errors
plt.colorbar(c3, ax=ax3)
ax3.set_title("Absolute Error |u - MMS| at t = T")
ax3.set_xlabel("x")
ax3.set_ylabel("y")

ax1 = axes[1, 0]
c1 = ax1.contourf(X, Y, v[:, :, -1], cmap="viridis")
plt.colorbar(c1, ax=ax1)
ax1.set_title("Numerical Solution at t = T")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# MMS (exact) solution at final time step
ax2 = axes[1, 1]
c2 = ax2.contourf(X, Y, v_exact[:, :, -1], cmap="viridis")
plt.colorbar(c2, ax=ax2)
ax2.set_title("MMS (Exact) Solution at t = T")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

# Absolute Error at final time step
ax3 = axes[1, 2]
c3 = ax3.contourf(X, Y, error_v[:, :, -1], cmap="inferno")  # Using 'inferno' to highlight errors
plt.colorbar(c3, ax=ax3)
ax3.set_title("Absolute Error |u - MMS| at t = T")
ax3.set_xlabel("x")
ax3.set_ylabel("y")

ax1 = axes[2, 0]
c1 = ax1.contourf(X, Y, p, cmap="viridis")
plt.colorbar(c1, ax=ax1)
ax1.set_title("Numerical Solution at t = T")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# MMS (exact) solution at final time step
ax2 = axes[2, 1]
c2 = ax2.contourf(X, Y, p_exact, cmap="viridis")
plt.colorbar(c2, ax=ax2)
ax2.set_title("MMS (Exact) Solution at t = T")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

# Absolute Error at final time step
ax3 = axes[2, 2]
c3 = ax3.contourf(X, Y, error_p, cmap="inferno")  # Using 'inferno' to highlight errors
plt.colorbar(c3, ax=ax3)
ax3.set_title("Absolute Error |u - MMS| at t = T")
ax3.set_xlabel("x")
ax3.set_ylabel("y")

plt.suptitle(f"Comparison at t = {T}", fontsize=14)
plt.tight_layout()
plt.show()

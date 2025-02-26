"""
===================================================================================
Title: 2D Nonlinear Convection Equation Solver with Method of Manufactured Solutions (MMS)
Author: Weichao Li
Date: 2025/02/21
Description:
    This script numerically solves the 2D nonlinear convection equation using the
    Method of Manufactured Solutions (MMS). The numerical results are compared
    against the exact MMS solution, and an error analysis is performed.

    The governing equations:
        ∂u/∂t + u * ∂u/∂x + v * ∂u/∂y = f_u(x, y, t)
        ∂v/∂t + u * ∂v/∂x + v * ∂v/∂y = f_v(x, y, t)

    where f_u and f_v are the source terms derived from the MMS solution:
        u(x,y,t) = exp(-t) * sin(pi*x) * cos(pi*y)
        v(x,y,t) = - exp(-t) * cos(pi*x) * sin(pi*y)

    The numerical method used is an explicit upwind finite difference scheme.

License: MIT License (if applicable)
===================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
nx, ny = 50, 50  # Grid points
nt = 200  # Time steps
T = 0.2  # Final time
Lx, Ly = 1.0, 1.0  # Domain size
cx, cy = 1.0, 1.0  # Convection speeds

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = T / (nt - 1)

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
t = np.linspace(0, T, nt)


# source term
def f_u(x_f, y_f, t_f):
    return (np.pi * np.cos(np.pi * x_f) + np.exp(t_f) * np.cos(np.pi * y_f)) * np.exp(-2 * t_f) * np.sin(np.pi * x_f)


def f_v(x_f, y_f, t_f):
    return (np.pi * np.cos(np.pi * y_f) + np.exp(t_f) * np.cos(np.pi * x_f)) * np.exp(-2 * t_f) * np.sin(np.pi * y_f)


# set the initial conditions
u = np.zeros((nx, ny, nt))
v = np.zeros((nx, ny, nt))

for i in range(nx):
    for j in range(ny):
        u[i, j, 0] = np.exp(-t[0]) * np.sin(np.pi * x[i]) * np.cos(np.pi * y[j])
        v[i, j, 0] = - np.exp(-t[0]) * np.cos(np.pi * x[i]) * np.sin(np.pi * y[j])

# set the boundary condition x = 0
for j in range(ny):
    for n in range(nt):
        u[0, j, n] = np.exp(-t[n]) * np.sin(np.pi * x[0]) * np.cos(np.pi * y[j])
        v[0, j, n] = - np.exp(-t[n]) * np.cos(np.pi * x[0]) * np.sin(np.pi * y[j])

# set the boundary condition y = 0
for i in range(nx):
    for n in range(nt):
        u[i, 0, n] = np.exp(-t[n]) * np.sin(np.pi * x[i]) * np.cos(np.pi * y[0])
        v[i, 0, n] = - np.exp(-t[n]) * np.cos(np.pi * x[i]) * np.sin(np.pi * y[0])

# main loop
for n in range(nt - 1):
    for i in range(1, nx):
        for j in range(1, ny):
            u[i, j, n + 1] = (u[i, j, n] - u[i, j, n] * dt / dx * (u[i, j, n] - u[i - 1, j, n])
                              - v[i, j, n] * dt / dx * (u[i, j, n] - u[i, j - 1, n]) + dt * f_u(x[i], y[j], t[n]))
            v[i, j, n + 1] = (v[i, j, n] - u[i, j, n] * dt / dx * (v[i, j, n] - v[i - 1, j, n])
                              - v[i, j, n] * dt / dx * (v[i, j, n] - v[i, j - 1, n]) + dt * f_v(x[i], y[j], t[n]))

# compute the exact solution
u_exact = np.zeros((nx, ny, nt))
v_exact = np.zeros((nx, ny, nt))
for n in range(nt):
    for i in range(nx):
        for j in range(ny):
            u_exact[i, j, n] = np.exp(-t[n]) * np.sin(np.pi * x[i]) * np.cos(np.pi * y[j])
            v_exact[i, j, n] = - np.exp(-t[n]) * np.cos(np.pi * x[i]) * np.sin(np.pi * y[j])

# plot the results
X, Y = np.meshgrid(x, y, indexing="ij")  # Generate meshgrid for plotting

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Numerical solution
ax1 = axes[0]
c1 = ax1.contourf(X, Y, u[..., -1], cmap="viridis")
plt.colorbar(c1, ax=ax1)
ax1.set_title("Numerical Solution")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# MMS (exact) solution
ax2 = axes[1]
c2 = ax2.contourf(X, Y, u_exact[..., -1], cmap="viridis")
plt.colorbar(c2, ax=ax2)
ax2.set_title("MMS (Exact) Solution")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

# Absolute Error Plot
error = u[..., -1] - u_exact[..., -1]
ax3 = axes[2]
c3 = ax3.contourf(X, Y, error, cmap="inferno")  # Using 'inferno' to highlight errors
plt.colorbar(c3, ax=ax3)
ax3.set_title("Absolute Error |u - MMS|")
ax3.set_xlabel("x")
ax3.set_ylabel("y")

plt.suptitle(f"Comparison at t = {T}", fontsize=14)
plt.tight_layout()
plt.show()

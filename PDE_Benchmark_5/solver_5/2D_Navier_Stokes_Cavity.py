# Required Libraries
import numpy as np
import matplotlib.pyplot as plt

# Define simulation parameters
L = 1
nx = ny = 51
nt = 500
Re = 100
dt = 0.01
rho = 1
nu = L / Re

# Discretize the equations
x = np.linspace(0, L, nx)
y = np.linspace(0, L, ny)
u = np.zeros((ny,nx))
v = np.zeros((ny,nx))
p = np.zeros((ny,nx))
residual_u = np.zeros((ny, nx))
residual_v = np.zeros_like(residual_u)
residual_p = np.zeros_like(residual_u)

# Iterate until convergence
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    pn = p.copy()

    # Boundary conditions
    u[:, -1] = 1  # Lid-driven top boundary
    u[0, :] = u[-1, :] = v[:, :] = 0  # No-slip condition

    # Solve Navier-Stokes equations
    u[1:-1, 1:-1] = un[1:-1, 1:-1] - dt / L * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2]) - dt / L * vn[1:-1, 1:-1] * \
                    (un[1:-1, 1:-1] - un[:-2, 1:-1]) + dt * nu * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) / L ** 2 + \
                    dt * nu * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) / L ** 2
    # Solve pressure-poisson equation
    it = 0
    pn = p.copy()
    p[1:-1, 1:-1] = (pn[1:-1, 2:] + pn[1:-1, :-2] + pn[2:, 1:-1] + pn[:-2, 1:-1]) / 4 - L ** 2 / 4 * (1 / dt * \
                    (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * L) + 1 / dt * (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * L))
                    
# Visualize velocity fields
X, Y = np.meshgrid(x, y)
plt.figure(figsize=(11,7))
plt.quiver(X[::3, ::3], Y[::3, ::3], u[::3, ::3], v[::3, ::3])
plt.show()

# Visualize pressure fields
plt.figure(figsize=(11,7))
plt.contourf(X, Y, p, alpha=0.5)
plt.colorbar()
plt.show()
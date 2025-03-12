import numpy as np
import matplotlib.pyplot as plt

# Simulation Parameters
L = 1.0   # size of grid 
nx, ny = 100, 100  # number of points in grid
nt = 500  # number of time steps
Re = 100.0  # Reynolds number
dt = 0.001  # time step size
rho = 1
nu = 1/Re  # kinematic viscosity

dx = L / (nx - 1)  # grid resolution
dy = L / (ny - 1)

# Discretize the domain
x = np.linspace(0.0, L, num=nx)
y = np.linspace(0.0, L, num=ny)

# Initialize the fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

# Applying the boundary conditions
u[:, -1] = 1.0  # drive cavity
u[-1, :] = 0.0  # no-slip boundary
v[-1, :] = 0.0  # no-slip boundary
p[:, :] = 0.0  # constant pressure

# Time loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    vn[-1, :] = 0.0

    # Solve the momentum equations
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] - dt / dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - 
                     dt / dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) + 
                     nu * dt / dx**2 * (un[1:-1,2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                     nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - dt / dx * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) - 
                     dt / dy * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) +
                     nu * dt / dx**2 * (vn[1:-1,2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                     nu * dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))

    # Solve the pressure Poisson equation
    for _ in range(ny):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         rho * dx**2 * dy**2 / (2 * (dx**2 + dy**2)) *
                         ((un[1:-1, 2:] - un[1:-1, :-2]) / (2 * dx) +
                          (vn[2:, 1:-1] - vn[:-2, 1:-1]) / (2 * dy))**2)

    # Boundary conditions
    p[:, -1] = p[:, -2]  # dp/dy=0 at x = 2
    p[0, :] = p[1, :]  # dp/dy=0 at y = 0
    p[:, 0] = p[:, 1]  # dp/dx=0 at x = 0

# Generating the quiver plot
plt.figure(figsize=(11, 7), dpi=100)
plt.quiver(x[::3], y[::3], u[::3, ::3], v[::3, ::3])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Generating the pressure contour plot
plt.figure(figsize=(8, 5), dpi=100)
plt.contourf(x, y, p, alpha=0.5, cmap='viridis')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
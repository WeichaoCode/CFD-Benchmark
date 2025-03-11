import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Lx, Ly = 1.0, 1.0  # domain size
T = 1.0  # total time
nx, ny, nt = 101, 101, 100  # number of grid points
dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # grid spacing
dt = T / (nt - 1)  # time step
cfl = dx / dt  # CFL condition

# Discretize space and time
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
t = np.linspace(0, T, nt)

u, v = np.zeros((ny, nx, nt)), np.zeros((ny, nx, nt))  # velocity fields

# Initialize velocity field
u[:, :, 0] = np.sin(2 * np.pi * x) * np.ones_like(y)[:, np.newaxis]
v[:, :, 0] = -2 * np.pi * y[:, np.newaxis] * np.cos(2 * np.pi * x)

# Finite difference scheme
for n in range(nt - 1):
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            u[j, i, n + 1] = (u[j, i, n] - dt / dx * u[j, i, n] * (u[j, i, n] - u[j, i - 1, n])
                              - dt / dy * v[j, i, n] * (u[j, i, n] - u[j - 1, i, n]))
            v[j, i, n + 1] = (v[j, i, n] - dt / dx * u[j, i, n] * (v[j, i, n] - v[j, i - 1, n])
                              - dt / dy * v[j, i, n] * (v[j, i, n] - v[j - 1, i, n]))

# Visualization
plt.figure(figsize=(6, 6))
plt.quiver(x, y, u[:, :, nt - 1], v[:, :, nt - 1])
plt.title('Velocity field at t = {0:.2f}'.format(T))
plt.xlabel('x')
plt.ylabel('y')
plt.show()
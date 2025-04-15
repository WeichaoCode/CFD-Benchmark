import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define parameters
nx = ny = 41
nt = 120
sigma = .0009
nu = 0.01
dx = dy = 2 / (nx - 1)
dt = sigma * dx * dy / nu

# Initialize variables
u = np.ones((ny, nx))
v = np.ones((ny, nx))
u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2
v[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2

# Time integration loop
for n in range(nt + 1):
    un = u.copy()
    vn = v.copy()
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] - dt / dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     dt / dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) +
                     nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                     nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - dt / dx * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     dt / dy * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) +
                     nu * dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                     nu * dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
    u[0, :], u[-1, :], u[:, 0], u[:, -1] = 1, 1, 1, 1
    v[0, :], v[-1, :], v[:, 0], v[:, -1] = 1, 1, 1, 1

# Visualization
fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.linspace(0, 2, nx), np.linspace(0, 2, ny))
ax.plot_surface(X, Y, u, cmap='viridis')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, v, cmap='viridis')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()

# Save the final velocity fields
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/u_2D_Burgers_Equation.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_6/results/prediction/v_2D_Burgers_Equation.npy', v)
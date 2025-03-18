import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define parameters
nx = ny = 101
nt = 80
sigma = 0.2
dx = dy = 2 / (nx - 1)
dt = sigma * dx

# Initialize variables
u = np.ones((ny, nx))
v = np.ones((ny, nx))
u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2
v[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2

# Time integration loop
for n in range(nt + 1):
    un = u.copy()
    vn = v.copy()
    u[1:, 1:] = (un[1:, 1:] - 
                 (un[1:, 1:] * dt / dx * (un[1:, 1:] - un[1:, :-1])) -
                 (vn[1:, 1:] * dt / dy * (un[1:, 1:] - un[:-1, 1:])))
    v[1:, 1:] = (vn[1:, 1:] -
                 (un[1:, 1:] * dt / dx * (vn[1:, 1:] - vn[1:, :-1])) -
                 (vn[1:, 1:] * dt / dy * (vn[1:, 1:] - vn[:-1, 1:])))
    
    # Apply boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Save the final velocity fields
np.save('u_final.npy', u)
np.save('v_final.npy', v)

# Visualization
fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(np.linspace(0, 2, nx), np.linspace(0, 2, ny))

ax.plot_surface(X, Y, u, cmap='viridis')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, v, cmap='viridis')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()
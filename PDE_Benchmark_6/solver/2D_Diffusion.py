import numpy as np
import matplotlib.pyplot as plt

# Define parameters
nx, ny = 31, 31
nt = 50
nu = .05
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .25
dt = sigma * dx * dy / nu
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

# Initialize variables
u = np.ones((ny, nx))
u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2

# Time integration loop
for n in range(nt + 1):
    un = u.copy()
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] + 
                     nu * dt / dx**2 * 
                     (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                     nu * dt / dy**2 * 
                     (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]))
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

# Visualization
plt.figure(figsize=(11, 7), dpi=100)
plt.contourf(x, y, u, alpha=0.5, cmap='viridis')
plt.colorbar()
plt.contour(x, y, u, cmap='viridis')
plt.show()

# Save the final solution
np.save('final_solution.npy', u)
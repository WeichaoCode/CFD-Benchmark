import numpy as np
import matplotlib.pyplot as plt

# Define grid parameters
nx = 81
ny = 81
nt = 100
c = 1.0
Lx = 2.0
Ly = 2.0
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
sigma = 0.2
dt = sigma * min(dx, dy) / c

# Initialize solution arrays
u = np.ones((ny, nx))  # u at time n
un = np.ones((ny, nx))  # u at time n+1

# Set initial condition
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    u[1:, 1:] = (un[1:, 1:] - (c * dt / dx * (un[1:, 1:] - un[1:, :-1])) -
                 (c * dt / dy * (un[1:, 1:] - un[:-1, 1:])))
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

# Save the solution to .npy file
np.save('solution.npy', u)

# Visualization
plt.figure(figsize=(8, 5))
plt.contourf(u, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='U')
plt.title('2D Linear Convection')
plt.show()
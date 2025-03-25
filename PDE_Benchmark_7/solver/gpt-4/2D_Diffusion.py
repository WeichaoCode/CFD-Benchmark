import numpy as np
import matplotlib.pyplot as plt

# Define parameters
nx, ny = 31, 31  # number of grid points
domain_size = 2.0  # domain size
nu = 0.05  # diffusion coefficient
nt = 50  # number of time steps
sigma = 0.25  # stability criteria
dx = dy = domain_size / (nx - 1)  # grid spacing
dt = sigma * dx * dy / nu  # time step size

# Initialize solution: u = 1 everywhere
u = np.ones((nx, ny))

# Set initial condition: u = 2 in the region 0.5 <= x, y <= 1
u[int(0.5 / dx):int(1 / dx + 1), int(0.5 / dy):int(1 / dy + 1)] = 2

# Time integration loop
for n in range(nt):
    un = u.copy()  # copy the existing values of u into un
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] + 
                     nu * dt / dx**2 * 
                     (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) +
                     nu * dt / dy**2 * 
                     (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]))
    # Apply boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

# Save the final solution in a .npy file
np.save('solution.npy', u)

# Visualization
x = np.linspace(0, domain_size, nx)
y = np.linspace(0, domain_size, ny)
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, u, cmap='viridis')
plt.colorbar()
plt.title('2D Diffusion after {} time steps'.format(nt))
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Lx, Ly = 2.0, 2.0  # domain size
nx, ny = 41, 41  # grid points
nt = 500  # time steps
nu = 0.1  # diffusion coefficient

# Discretize space and time
dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # grid spacing
dt = dx * dy / (4 * nu)  # time step for stability

# Initialize field u
u = np.zeros((ny, nx))
u_center = int(ny / 2), int(nx / 2)  # locate the center
u[u_center[0]-5:u_center[0]+5, u_center[1]-5:u_center[1]+5] = 1.0  # small blob at the center

# Initialize "old" field for iteration
un = np.empty_like(u)

# Diffusion iteration
for n in range(nt):
    un = u.copy()
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] +
                     nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                     nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]))
    # Apply boundary conditions
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0

    if n % 50 == 0:  # only plot every 50th time step
        plt.imshow(u, cmap='hot', origin='lower')
        plt.colorbar()
        plt.title('2D Diffusion at time step {}'.format(n))
        plt.show()
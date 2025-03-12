import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Lx, Ly = 2., 2.
T = 2.0
nx, ny, nt = 101, 101, 100
nu = 0.1

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = dx*dy*nu

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Initialize field
u = np.empty((ny, nx))
u_new = np.empty((ny, nx))

u.fill(0.5)
u[int(.5 / dx):int(1 / dx + 1),int(.5 / dx):int(1 / dx + 1)] = 2

# Iterative solve
for n in range(nt + 1):
    u_new = u.copy()
    u[1:-1, 1:-1] = (u_new[1:-1, 1:-1]
                     + nu * dt / dx**2 * 
                     (u_new[1:-1, 2:] - 2 * u_new[1:-1, 1:-1] + u_new[1:-1, :-2])
                     + nu * dt / dy**2 * 
                     (u_new[2:, 1:-1] - 2 * u_new[1:-1, 1:-1] + u_new[:-2, 1:-1]))
    
    # Apply boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    if n % 10 == 0:
        plt.figure()
        plt.imshow(u, cmap='hot', extent=[0, Lx, 0, Ly])
        plt.colorbar()
        plt.show()
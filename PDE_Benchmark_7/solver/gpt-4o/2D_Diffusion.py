import numpy as np
import matplotlib.pyplot as plt

# Define Parameters
nx, ny = 31, 31  # number of grid points
Lx, Ly = 2.0, 2.0  # domain size
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

nu = 0.1  # diffusion coefficient
sigma = 0.25
dt = sigma * dx * dy / nu  # time step size
nt = 50  # number of time steps

# Initialize Variables
u = np.ones((ny, nx))  # initialize the grid with 1s
# apply initial condition: set u = 2 in the region 0.5 <= x <= 1 and 0.5 <= y <= 1
u[int(0.5 / dy):int(1.0 / dy + 1), int(0.5 / dx):int(1.0 / dx + 1)] = 2

# Time integration loop
for n in range(nt):
    un = u.copy()  # copy the current grid state
    # Update the grid using the finite difference method
    u[1:-1, 1:-1] = (
        un[1:-1, 1:-1]
        + nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
        + nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])
    )
    # Apply boundary conditions
    u[:, 0] = 1
    u[:, -1] = 1
    u[0, :] = 1
    u[-1, :] = 1

# Visualization
plt.figure(figsize=(7, 5))
plt.contourf(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), u, cmap=plt.cm.viridis)
plt.colorbar(label='Concentration')
plt.title('2D Diffusion at final time step')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/u_2D_Diffusion.npy', u)
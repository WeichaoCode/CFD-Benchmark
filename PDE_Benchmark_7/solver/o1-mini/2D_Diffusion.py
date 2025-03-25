import numpy as np
import matplotlib.pyplot as plt

# Define parameters
nx = ny = 31
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
nt = 50
sigma = 0.25
nu = 0.1
dt = sigma * dx * dy / nu

# Initialize u
u = np.ones((ny, nx))

# Set initial condition: u=2 in the region 0.5 <= x, y <= 1
X, Y = np.meshgrid(x, y)
u[np.where((X >= 0.5) & (X <= 1) & (Y >= 0.5) & (Y <= 1))] = 2

# Time integration using Explicit Euler Method
for n in range(nt):
    u_old = u.copy()
    
    # Compute the diffusion terms using central differences
    u[1:-1, 1:-1] = u_old[1:-1, 1:-1] + \
        nu * dt / dx**2 * (u_old[1:-1, 2:] - 2 * u_old[1:-1, 1:-1] + u_old[1:-1, :-2]) + \
        nu * dt / dy**2 * (u_old[2:, 1:-1] - 2 * u_old[1:-1, 1:-1] + u_old[:-2, 1:-1])
    
    # Apply Dirichlet boundary conditions (u=1 at boundaries)
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    # Visualization at specific time steps
    if n % 10 == 0:
        plt.figure(figsize=(6,5))
        cp = plt.contourf(X, Y, u, alpha=0.8, cmap='viridis')
        plt.colorbar(cp)
        plt.title(f'Diffusion at time step {n}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

# Save the final solution to a .npy file
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/u_2D_Diffusion.npy', u)
import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 1.0        # Convection speed
nx = 81        # Number of grid points in x
ny = 81        # Number of grid points in y
Lx = 2.0       # Domain length in x
Ly = 2.0       # Domain length in y
nt = 100       # Number of time steps
sigma = 0.2    # Stability parameter

# Spatial discretization
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Time step based on CFL condition
dt = sigma * min(dx, dy) / c

# Initialize u to 1 everywhere
u = np.ones((ny, nx))

# Apply initial condition: u=2 for 0.5 <= x <=1 and 0.5 <= y <=1
u[np.where((X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0))] = 2.0

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    # Update interior points
    u[1:, 1:] = (un[1:, 1:] -
                 (c * dt / dx) * (un[1:, 1:] - un[1:, 0:-1]) -
                 (c * dt / dy) * (un[1:, 1:] - un[0:-1, 1:]))
    # Apply boundary conditions: u=1 on all boundaries
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

# Visualization using contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, u, levels=50, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'2D Linear Convection after {nt} Time Steps')
plt.colorbar(contour)
plt.show()

# Save the final solution to a .npy file
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/results/prediction/o1-mini/u_2D_Linear_Convection.npy', u)
import numpy as np
import matplotlib.pyplot as plt

# Define constants
nx, ny = 81, 81    # Number of grid points
Lx, Ly = 2.0, 2.0  # Domain size
c = 1.0            # Convection speed
σ = 0.2            # Stability parameter
nt = 100           # Number of time steps

# Calculate spatial and temporal parameters
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
dt = σ * min(dx, dy) / c

# Grid setup
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initial condition
u = np.ones((ny, nx))  # Initialize u with boundary conditions
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Temporary array for time-stepping
u_n = np.ones((ny, nx))

# Time-stepping loop
for n in range(nt):
    u_n[:, :] = u[:]  # Copy current state into temporary array

    # Update the solution
    u[1:, 1:] = (u_n[1:, 1:]
                 - (c * dt / dx * (u_n[1:, 1:] - u_n[1:, :-1]))
                 - (c * dt / dy * (u_n[1:, 1:] - u_n[:-1, 1:])))

    # Enforce boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/u_2D_Linear_Convection.npy', u)

# Plot the solution
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, u, cmap='viridis')
plt.colorbar(label='u')
plt.title('2D Linear Convection at final time step')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 5.0, 4.0  # Domain size
dx, dy = 0.05, 0.05  # Grid spacing
nx, ny = int(Lx/dx) + 1, int(Ly/dy) + 1  # Number of grid points
omega = 1.5  # Relaxation factor
tolerance = 1e-4  # Convergence criterion

# Initialize temperature field
T = np.zeros((ny, nx))

# Apply Dirichlet boundary conditions
T[:, 0] = 10.0  # Left boundary
T[:, -1] = 40.0  # Right boundary
T[0, :] = 20.0  # Bottom boundary
T[-1, :] = 0.0  # Top boundary

# SOR method
beta = dx / dy
converged = False
iteration = 0

while not converged:
    T_old = T.copy()
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T[j, i] = omega * (T_old[j, i+1] + T[j, i-1] + beta**2 * (T_old[j+1, i] + T[j-1, i])) / (2 * (1 + beta**2)) + (1 - omega) * T_old[j, i]
    
    # Enforce boundary conditions
    T[:, 0] = 10.0
    T[:, -1] = 40.0
    T[0, :] = 20.0
    T[-1, :] = 0.0

    # Check for convergence
    max_diff = np.max(np.abs(T - T_old))
    if max_diff < tolerance:
        converged = True
    iteration += 1

# Save the final temperature field
np.save('temperature_distribution.npy', T)

# Plot the temperature distribution
plt.figure(figsize=(8, 6))
plt.contourf(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), T, 20, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
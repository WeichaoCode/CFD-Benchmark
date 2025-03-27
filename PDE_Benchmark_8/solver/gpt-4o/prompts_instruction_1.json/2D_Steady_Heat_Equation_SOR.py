import numpy as np

# Parameters
Lx, Ly = 5.0, 4.0  # Domain size
dx, dy = 0.05, 0.05  # Grid spacing
nx, ny = int(Lx/dx) + 1, int(Ly/dy) + 1  # Number of grid points
omega = 1.5  # Relaxation factor
tolerance = 1e-4  # Convergence criterion

# Initialize temperature field
T = np.zeros((ny, nx))

# Apply boundary conditions
T[:, 0] = 10.0  # Left boundary
T[:, -1] = 40.0  # Right boundary
T[0, :] = 0.0  # Top boundary
T[-1, :] = 20.0  # Bottom boundary

# SOR method
beta = dx / dy
converged = False

while not converged:
    T_old = T.copy()
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T[j, i] = omega * (T_old[j, i+1] + T[j, i-1] + beta**2 * (T_old[j+1, i] + T[j-1, i])) / (2 * (1 + beta**2)) + (1 - omega) * T_old[j, i]
    
    # Enforce boundary conditions
    T[:, 0] = 10.0
    T[:, -1] = 40.0
    T[0, :] = 0.0
    T[-1, :] = 20.0
    
    # Check for convergence
    max_diff = np.max(np.abs(T - T_old))
    if max_diff < tolerance:
        converged = True

# Save the final solution
np.save('temperature_distribution.npy', T)
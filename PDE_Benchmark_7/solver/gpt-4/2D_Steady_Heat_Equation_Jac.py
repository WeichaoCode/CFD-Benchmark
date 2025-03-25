import numpy as np
import matplotlib.pyplot as plt

# Define the grid parameters
dx = 0.05
dy = 0.05
nx = int(5.0/dx) + 1
ny = int(4.0/dy) + 1
beta = dx / dy

# Initialize the temperature field and the residual
T = np.zeros((nx, ny))
residual = np.ones((nx, ny))

# Set the boundary conditions
T[:, 0] = 10.0  # left boundary
T[0, :] = 0.0   # top boundary
T[:, -1] = 40.0 # right boundary
T[-1, :] = 20.0 # bottom boundary

# Iterate until convergence
tolerance = 1e-6
while np.max(residual) > tolerance:
    T_old = T.copy()
    T[1:-1, 1:-1] = (T_old[2:, 1:-1] + T_old[:-2, 1:-1] + beta**2 * (T_old[1:-1, 2:] + T_old[1:-1, :-2])) / (2 * (1 + beta**2))
    residual = np.abs(T - T_old)

# Visualize the steady-state temperature distribution
plt.figure(figsize=(8, 6))
plt.contourf(T, cmap='hot')
plt.colorbar(label='Temperature (C)')
plt.title('Steady-State Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Save the final temperature field
np.save('temperature.npy', T)
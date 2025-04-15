import numpy as np
import matplotlib.pyplot as plt

# Function to update the domain with the new temperature distribution
def update_domain(T, dx, dy, tolerance=1e-4):
    deltaT = tolerance
    while deltaT >= tolerance:
        T_old = T.copy()
        T[1:-1, 1:-1] = ((dx**2 * (T_old[:-2, 1:-1] + T_old[2:, 1:-1]) +
                          dy**2 * (T_old[1:-1, :-2] + T_old[1:-1, 2:])) /
                         (2.0 * (dx**2 + dy**2)))
        deltaT = np.max(np.abs(T - T_old))
    return T

# Function for setting up initial domain
def setup_domain(x_len, y_len, nx, ny):
    dx = x_len / (nx-1)
    dy = y_len / (ny-1)
    T = np.zeros((ny, nx))  # Initialize domain with zeros

    # Set boundary conditions
    T[:, 0] = 10     # Left boundary
    T[:, -1] = 40    # Right boundary
    T[0, :] = 0      # Top boundary
    T[-1, :] = 20    # Bottom boundary

    return T, dx, dy

# Define size of domain and resolution
x_len, y_len = 5, 4
nx, ny = 100, 100  # Number of nodes

T, dx, dy = setup_domain(x_len, y_len, nx, ny)
T = update_domain(T, dx, dy)

# Plot temperature distribution
plt.figure(figsize=(10, 6))
plt.contourf(T, levels=50, cmap='jet')
plt.title('Steady-state heat distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Temperature (°C)')
plt.show()
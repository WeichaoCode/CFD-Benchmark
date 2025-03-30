import numpy as np
import matplotlib.pyplot as plt

# Parameters
x_min, x_max = 0, 5
y_min, y_max = 0, 4
dx, dy = 0.05, 0.05
nx, ny = 101, 81
omega = 1.5
beta = dx / dy
convergence_threshold = 1e-4
max_iterations = 10000

# Initialize temperature field
T = np.zeros((ny, nx))

# Apply Dirichlet boundary conditions
T[:, 0] = 10.0      # Left boundary
T[:, -1] = 40.0     # Right boundary
T[0, :] = 0.0       # Top boundary
T[-1, :] = 20.0     # Bottom boundary

# SOR Iteration
for iteration in range(max_iterations):
    T_old = T.copy()
    max_diff = 0.0
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T_new = omega * (T[j, i+1] + T[j, i-1] + beta**2 * (T[j+1, i] + T[j-1, i])) / (2 * (1 + beta**2)) + (1 - omega) * T[j, i]
            diff = abs(T_new - T[j, i])
            if diff > max_diff:
                max_diff = diff
            T[j, i] = T_new
    # Re-apply Dirichlet boundary conditions
    T[:, 0] = 10.0
    T[:, -1] = 40.0
    T[0, :] = 0.0
    T[-1, :] = 20.0
    if max_diff < convergence_threshold:
        break

# Save the final temperature field
np.save('T.npy', T)

# Generate contour plot
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, T, 50, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title('Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('temperature_contour.png')
plt.close()
import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
x_start, x_end, dx = 0, 5, 0.05
y_start, y_end, dy = 0, 4, 0.05
nx, ny = 101, 81

# Initialize temperature field
T = np.zeros((nx, ny))

# Apply Dirichlet boundary conditions
T[:, 0] = 10      # Left boundary
T[:, -1] = 40     # Right boundary
T[0, :] = 20      # Bottom boundary
T[-1, :] = 0      # Top boundary

# Parameters for Gauss-Seidel
beta = dx / dy
tolerance = 1e-4
max_iterations = 10000
iteration = 0
max_diff = tolerance + 1

# Gauss-Seidel iteration
while max_diff > tolerance and iteration < max_iterations:
    max_diff = 0
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            T_new = (T[i+1, j] + T[i-1, j] + beta**2 * (T[i, j+1] + T[i, j-1])) / (2 * (1 + beta**2))
            diff = abs(T_new - T[i, j])
            if diff > max_diff:
                max_diff = diff
            T[i, j] = T_new
    # Re-apply boundary conditions
    T[:, 0] = 10
    T[:, -1] = 40
    T[0, :] = 20
    T[-1, :] = 0
    iteration += 1

# Save the final temperature field
np.save('T.npy', T)

# Create contour plot
x = np.linspace(x_start, x_end, nx)
y = np.linspace(y_start, y_end, ny)
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, T.T, 50, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Temperature Distribution')
plt.show()
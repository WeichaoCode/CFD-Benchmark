import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
width = 5
height = 4
dx = 0.05
dy = 0.05
nx = 101
ny = 81

# Compute beta
beta = dx / dy

# Initialize temperature field
T = np.zeros((ny, nx))

# Apply Dirichlet boundary conditions
T[:, 0] = 10    # Left boundary
T[:, -1] = 40    # Right boundary
T[0, :] = 20     # Bottom boundary
T[-1, :] = 0     # Top boundary

# Convergence parameters
tolerance = 1e-4
max_diff = tolerance + 1
iteration = 0

# Gauss-Seidel iteration
while max_diff > tolerance:
    max_diff = 0
    for i in range(1, ny-1):
        for j in range(1, nx-1):
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

# Generate contour plot
x = np.linspace(0, width, nx)
y = np.linspace(0, height, ny)
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, T, 50, cmap='jet')
plt.colorbar(label='Temperature (°C)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Temperature Distribution')
plt.show()
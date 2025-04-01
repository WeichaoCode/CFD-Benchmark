import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
width = 5.0
height = 4.0
dx = 0.05
dy = 0.05
nx = 101
ny = 81

# Numerical parameters
beta = dx / dy
convergence_threshold = 1e-4

# Initialize temperature field
T = np.zeros((ny, nx))
# Apply Dirichlet boundary conditions
T[:, 0] = 10.0        # Left boundary (AB, x=0)
T[:, -1] = 40.0        # Right boundary (EF, x=5)
T[0, :] = 20.0         # Bottom boundary (G, y=0)
T[-1, :] = 0.0         # Top boundary (CD, y=4)

# Initialize variables for iteration
T_new = T.copy()
residual = np.inf

# Jacobi iteration
while residual > convergence_threshold:
    T_new[1:-1, 1:-1] = (T[2:, 1:-1] + T[:-2, 1:-1] + beta**2 * (T[1:-1, 2:] + T[1:-1, :-2])) / (2 * (1 + beta**2))
    # Apply Dirichlet boundary conditions
    T_new[:, 0] = 10.0
    T_new[:, -1] = 40.0
    T_new[0, :] = 20.0
    T_new[-1, :] = 0.0
    # Compute residual
    residual = np.max(np.abs(T_new - T))
    # Update temperature field
    T, T_new = T_new, T

# Save the final temperature field
np.save('T.npy', T)

# Generate contour plot
x = np.linspace(0, width, nx)
y = np.linspace(0, height, ny)
X, Y = np.meshgrid(x, y)

plt.contourf(X, Y, T, levels=50, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Temperature Distribution')
plt.show()
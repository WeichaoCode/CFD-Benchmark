import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
width = 5.0
height = 4.0
dx = 0.05
dy = 0.05
nx = 101
ny = 81

# Grid
x = np.linspace(0, width, nx)
y = np.linspace(0, height, ny)
T = np.zeros((ny, nx))

# Boundary conditions
T[:, 0] = 10.0       # Left boundary (x=0)
T[:, -1] = 40.0      # Right boundary (x=5)
T[0, :] = 20.0       # Bottom boundary (y=0)
T[-1, :] = 0.0       # Top boundary (y=4)

# SOR parameters
omega = 1.5
beta = dx / dy
tolerance = 1e-4
max_iterations = 10000
residual = np.inf
iteration = 0

# SOR iterative solver
while residual > tolerance and iteration < max_iterations:
    residual = 0.0
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T_old = T[j, i]
            T_new = omega * (T[j, i+1] + T[j, i-1] + beta**2 * (T[j+1, i] + T[j-1, i])) / (2 * (1 + beta**2)) + (1 - omega) * T_old
            diff = abs(T_new - T_old)
            if diff > residual:
                residual = diff
            T[j, i] = T_new
    # Re-apply boundary conditions
    T[:, 0] = 10.0
    T[:, -1] = 40.0
    T[0, :] = 20.0
    T[-1, :] = 0.0
    iteration += 1

# Save the final temperature field
np.save('T.npy', T)

# Generate contour plot
X, Y = np.meshgrid(x, y)
plt.figure(figsize=(8,6))
contour = plt.contourf(X, Y, T, 50, cmap='hot')
plt.colorbar(contour)
plt.title('Temperature Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('temperature_contour.png')
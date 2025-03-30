import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 1.0
nx, ny = 31, 31
nt = 50
sigma = 0.25
dx = 2.0 / (nx - 1)
dy = 2.0 / (ny - 1)
dt = sigma * dx * dy / nu

# Create grid
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# Initialize u
u = np.ones((ny, nx))
u[np.where((X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0))] = 2.0

# Time-stepping
for _ in range(nt):
    un = u.copy()
    u[1:-1, 1:-1] = (
        un[1:-1, 1:-1]
        + nu * dt * (
            (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) / dx**2
            + (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]) / dy**2
        )
    )
    # Apply Dirichlet boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

# Visualization
plt.figure(figsize=(7,6))
contour = plt.contourf(X, Y, u, alpha=0.5, cmap='viridis')
plt.contour(X, Y, u, cmap='viridis')
plt.colorbar(contour)
plt.title('2D Diffusion at Final Time Step')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Save the final solution
np.save('u.npy', u)
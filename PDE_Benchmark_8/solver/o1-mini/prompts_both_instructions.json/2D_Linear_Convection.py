import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 81, 81
x_start, x_end = 0.0, 2.0
y_start, y_end = 0.0, 2.0
dx = (x_end - x_start) / (nx - 1)
dy = (y_end - y_start) / (ny - 1)
nt = 100
sigma = 0.2
c = 1.0
dt = sigma * min(dx, dy) / c

# Initialize u
u = np.ones((nx, ny))
x = np.linspace(x_start, x_end, nx)
y = np.linspace(y_start, y_end, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
mask = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
u[mask] = 2.0

# Time-stepping
for _ in range(nt):
    u_new = np.copy(u)
    u_new[1:, 1:] = u[1:, 1:] - c * dt / dx * (u[1:, 1:] - u[:-1, 1:]) - c * dt / dy * (u[1:, 1:] - u[1:, :-1])
    # Enforce Dirichlet boundary conditions
    u_new[0, :] = 1.0
    u_new[-1, :] = 1.0
    u_new[:, 0] = 1.0
    u_new[:, -1] = 1.0
    u = u_new

# Save the final solution
np.save('u.npy', u)

# Visualization
plt.figure(figsize=(8,6))
contour = plt.contourf(X, Y, u, 20, cmap='viridis')
plt.colorbar(contour)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Final Solution at t = {:.3f}'.format(nt * dt))
plt.show()
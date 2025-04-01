import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 81, 81
nt = 100
c = 1.0
dx = 2.0 / (nx - 1)
dy = 2.0 / (ny - 1)
sigma = 0.2
dt = sigma * min(dx, dy) / c

# Initialize grid
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# Initialize u with initial conditions
u = np.ones((ny, nx))
u[(X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)] = 2.0

# Time-stepping
for _ in range(nt):
    u_new = u.copy()
    u_new[1:, 1:] = (
        u[1:, 1:]
        - c * dt / dx * (u[1:, 1:] - u[1:, :-1])
        - c * dt / dy * (u[1:, 1:] - u[:-1, 1:])
    )
    # Apply Dirichlet boundary conditions
    u_new[0, :] = 1.0
    u_new[-1, :] = 1.0
    u_new[:, 0] = 1.0
    u_new[:, -1] = 1.0
    u = u_new

# Save final solution
np.save('u.npy', u)

# Visualization
plt.contourf(X, Y, u, levels=50, cmap='viridis')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Final Solution at t = {:.3f}'.format(nt * dt))
plt.show()
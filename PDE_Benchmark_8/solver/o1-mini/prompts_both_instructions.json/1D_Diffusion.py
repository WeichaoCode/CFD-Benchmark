import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 41
nt = 20
nu = 0.3
sigma = 0.2
dx = 2 / (nx - 1)
dt = sigma * dx**2 / nu

# Spatial grid
x = np.linspace(0, 2, nx)

# Initial condition
u = np.ones(nx)
u[np.where((x >= 0.5) & (x <= 1.0))] = 2

# Time-stepping
for _ in range(nt):
    u_new = u.copy()
    u_new[1:-1] = u[1:-1] + nu * dt / dx**2 * (u[2:] - 2*u[1:-1] + u[:-2])
    # Apply Dirichlet boundary conditions
    u_new[0] = 1
    u_new[-1] = 0
    u = u_new

# Plot final solution
plt.plot(x, u, label='Final Solution')
plt.xlabel('x')
plt.ylabel('u')
plt.title('1D Diffusion Equation Final Solution')
plt.legend()
plt.show()

# Save the final solution
np.save('u.npy', u)
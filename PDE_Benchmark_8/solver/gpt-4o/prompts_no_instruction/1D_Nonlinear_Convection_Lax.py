import numpy as np
import math

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
dx = dt / nu
nx = math.ceil(L / dx)
T = 500

# Discretize the spatial domain
x = np.linspace(0, L, nx, endpoint=False)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Lax method for time integration
u_new = np.zeros_like(u)
for n in range(T):
    u_new[1:-1] = 0.5 * (u[2:] + u[:-2]) - dt / (2 * dx) * (u[2:]**2 - u[:-2]**2) / 2
    u_new[0] = 0.5 * (u[1] + u[-1]) - dt / (2 * dx) * (u[1]**2 - u[-1]**2) / 2
    u_new[-1] = u_new[0]  # Periodic boundary condition
    u[:] = u_new

# Save the final solution
save_values = ['u']
np.save(save_values[0] + '.npy', u)
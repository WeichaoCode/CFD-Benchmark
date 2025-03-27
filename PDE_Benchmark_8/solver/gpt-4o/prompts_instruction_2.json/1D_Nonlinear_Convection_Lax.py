import numpy as np

# Parameters
L = 2 * np.pi
nu = 0.5
dt = 0.01
dx = dt / nu
T = 500

# Discretize the spatial domain
x = np.linspace(0, L, int(np.ceil(L / dx)), endpoint=False)
nx = len(x)

# Initial condition
u = np.sin(x) + 0.5 * np.sin(0.5 * x)

# Lax method for time integration
def lax_method(u, dt, dx, nx):
    u_new = np.zeros_like(u)
    for i in range(1, nx - 1):
        u_new[i] = 0.5 * (u[i+1] + u[i-1]) - dt / (2 * dx) * (u[i+1]**2 / 2 - u[i-1]**2 / 2)
    # Apply periodic boundary conditions
    u_new[0] = 0.5 * (u[1] + u[-1]) - dt / (2 * dx) * (u[1]**2 / 2 - u[-1]**2 / 2)
    u_new[-1] = u_new[0]
    return u_new

# Time-stepping loop
for _ in range(T):
    u = lax_method(u, dt, dx, nx)

# Save the final solution
np.save('final_solution.npy', u)
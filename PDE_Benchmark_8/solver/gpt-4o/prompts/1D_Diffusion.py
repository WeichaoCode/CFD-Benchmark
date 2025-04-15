import numpy as np

# Parameters
nu = 0.3  # diffusion coefficient
L = 2.0   # length of the domain
T = 0.0333  # total time
nx = 101  # number of spatial points
nt = 100  # number of time steps
dx = L / (nx - 1)  # spatial step size
dt = T / nt  # time step size

# Stability condition for explicit method
assert nu * dt / dx**2 <= 0.5, "Stability condition violated!"

# Initial condition
u = np.ones(nx)
u[int(0.5 / dx):int(1 / dx) + 1] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    for i in range(1, nx - 1):
        u[i] = un[i] + nu * dt / dx**2 * (un[i + 1] - 2 * un[i] + un[i - 1])

# Save the final solution
np.save('u', u)
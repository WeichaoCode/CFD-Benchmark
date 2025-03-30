import numpy as np

# Parameters
nu = 0.07
nx = 101
nt = 100
L = 2 * np.pi
dx = L / (nx - 1)
dt = dx * nu

# Spatial grid
x = np.linspace(0, L, nx)

# Initial condition
phi = np.exp(-x**2 / (4 * nu)) + np.exp(-(x - L)**2 / (4 * nu))
u = -2 * nu / phi * (-x / (2 * nu) * np.exp(-x**2 / (4 * nu)) - (x - L) / (2 * nu) * np.exp(-(x - L)**2 / (4 * nu))) + 4

# Time-stepping
for _ in range(nt):
    un = u.copy()

    # Periodic boundary conditions
    u_left = np.roll(un, 1)
    u_right = np.roll(un, -1)

    # Convection term (upwind scheme)
    du_dx = (un - u_left) / dx

    # Diffusion term (central difference)
    d2u_dx2 = (u_right - 2 * un + u_left) / dx**2

    # Update
    u = un - dt * un * du_dx + nu * dt * d2u_dx2

# Save the final velocity field
np.save('u.npy', u)
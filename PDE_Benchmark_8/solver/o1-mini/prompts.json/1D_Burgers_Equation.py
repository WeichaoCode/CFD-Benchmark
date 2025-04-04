import numpy as np

# Parameters
nu = 0.07
x_start = 0.0
x_end = 2.0 * np.pi
t_start = 0.0
t_end = 0.14 * np.pi

# Discretization
N = 256
x = np.linspace(x_start, x_end, N, endpoint=False)
dx = (x_end - x_start) / N

# Time step based on CFL condition
dt = 0.001
num_steps = int((t_end - t_start) / dt)

# Initial condition
phi = np.exp(-x**2 / (4 * nu)) + np.exp(-(x - 2 * np.pi)**2 / (4 * nu))
dphi_dx = (np.roll(phi, -1) - np.roll(phi, 1)) / (2 * dx)
u = -2 * nu / phi * dphi_dx + 4

# Time-stepping
for _ in range(num_steps):
    # Compute derivatives
    du_dx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    d2u_dx2 = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
    # Update u
    u = u - dt * u * du_dx + dt * nu * d2u_dx2

# Save the final solution
np.save('u.npy', u)
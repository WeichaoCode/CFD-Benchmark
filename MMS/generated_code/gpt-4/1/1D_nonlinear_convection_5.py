import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 2.0  # length of domain
T = 2.0  # time of simulation
nx = 101  # number of spatial points in grid
nt = 101  # number of time steps
dx = L / (nx - 1)  # spatial grid size
dt = T / (nt - 1)  # time step size

# Grids
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# Initial condition
u0 = np.sin(np.pi * x)

# Solution array
u = np.zeros((nt, nx))
u[0, :] = u0

# Time-stepping loop
for n in range(nt - 1):
    for i in range(1, nx - 1):
        # Lax-Friedrichs method
        u[n + 1, i] = 0.5 * (u[n, i + 1] + u[n, i - 1]) - dt / (2 * dx) * (u[n, i] ** 2 - u[n, i - 1] ** 2)
    # Boundary conditions
    u[n + 1, 0] = u[n + 1, -1] = 0

# Plotting
plt.figure(figsize=(10, 6))
plt.title("1D Nonlinear Convection - Lax-Friedrichs")
plt.plot(x, u[0, :], label='t = 0')
plt.plot(x, u[nt // 4, :], label='t = T/4')
plt.plot(x, u[nt // 2, :], label='t = T/2')
plt.plot(x, u[-1, :], label='t = T')
plt.legend()
plt.show()
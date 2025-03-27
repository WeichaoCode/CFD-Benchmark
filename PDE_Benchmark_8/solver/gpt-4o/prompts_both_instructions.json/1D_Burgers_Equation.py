import numpy as np

# Parameters
nx = 101
nt = 100
nu = 0.07
x = np.linspace(0, 2 * np.pi, nx)
dx = 2 * np.pi / (nx - 1)
dt = dx * nu

# Initial condition
phi = np.exp(-(x**2) / (4 * nu)) + np.exp(-((x - 2 * np.pi)**2) / (4 * nu))
u = -2 * nu * (np.gradient(phi, dx)) / phi + 4

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    # Upwind scheme for convection term
    u[1:-1] = (un[1:-1] - un[1:-1] * dt / dx * (un[1:-1] - un[:-2]) +
               nu * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[:-2]))
    
    # Periodic boundary conditions
    u[0] = (un[0] - un[0] * dt / dx * (un[0] - un[-2]) +
            nu * dt / dx**2 * (un[1] - 2 * un[0] + un[-2]))
    u[-1] = u[0]

# Save the final solution to a .npy file
np.save('burgers_final_solution.npy', u)
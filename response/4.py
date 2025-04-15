import numpy as np
import matplotlib.pyplot as plt

# Parameters
nu = 0.07  # Viscosity
L = 2 * np.pi  # Length of the domain
nx = 101  # Number of spatial points
dx = L / (nx - 1)  # Spatial step size
dt = 0.002  # Time step size
nt = 500  # Number of time steps

# Spatial grid
x = np.linspace(0, L, nx)

# Initial condition
phi = np.exp(-x ** 2 / (4 * nu)) + np.exp(-((x - 2 * np.pi) ** 2) / (4 * nu))
u = -2 * nu / phi * np.gradient(phi, dx) + 4

# Initialize u for the next time step
u_next = np.zeros_like(u)

# Time-stepping loop (Forward Euler method)
for n in range(nt):
    u_next[1:-1] = (
            u[1:-1]
            - dt * u[1:-1] * (u[2:] - u[:-2]) / (2 * dx)  # Convection term
            + nu * dt * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2  # Diffusion term
    )
    # Periodic boundary conditions
    u_next[0] = u_next[-2]
    u_next[-1] = u_next[1]

    # Update u for the next iteration
    u[:] = u_next

##############################################
# The following lines are used to print output
##############################################
print(f"Solution of u is {np.array(u)}.")

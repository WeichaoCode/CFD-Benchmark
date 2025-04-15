import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 41  # Number of spatial grid points
L = 2.0  # Length of domain
dx = L / (nx - 1)  # Grid spacing
nu = 0.3  # Viscosity coefficient
dt = 0.0025  # Time step size (chosen based on stability)
nt = 50  # Number of time steps

# Discretized space grid
x = np.linspace(0, L, nx)

# Initial condition
u = np.ones(nx)  # Initialize with u=1 everywhere
u[(x >= 0.5) & (x <= 1)] = 2  # Set u=2 for 0.5 ≤ x ≤ 1

# Time stepping loop using the Central Difference scheme for second derivative
for t in range(nt):
    u_new = np.copy(u)  # Copy current values
    for i in range(1, nx - 1):  # Central Difference in space
        u_new[i] = u[i] + nu * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
    u = u_new  # Update solution

##############################################
# The following lines are used to print output
##############################################
print(f"Solution of u is {np.array(u)}.")



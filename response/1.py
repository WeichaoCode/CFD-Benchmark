import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 40  # Number of spatial grid points
L = 2.0  # Length of domain
dx = L / (nx - 1)  # Grid spacing
nt = 25  # Number of time steps
dt = 0.025  # Time step size
c = 1.0  # Wave speed

# Discretized space grid
x = np.linspace(0, L, nx)

# Initial condition
u = np.ones(nx)  # Initialize with u=1 everywhere
u[(x >= 0.5) & (x <= 1)] = 2  # Set u=2 for 0.5 <= x <= 1

# Plot initial condition
plt.figure(figsize=(8, 5))
plt.plot(x, u, label='Initial Condition')

# Time stepping loop using forward-time and backward-space scheme
for t in range(nt):
    u_new = np.copy(u)  # Copy current values
    for i in range(1, nx):  # Backward difference in space
        u_new[i] = u[i] - c * dt / dx * (u[i] - u[i-1])
    u = u_new  # Update solution
######################################################
# The following lines are added to test the output
######################################################
print(f"Solution of u is {np.array(u)}.")
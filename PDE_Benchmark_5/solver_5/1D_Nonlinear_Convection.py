import numpy as np
import matplotlib.pyplot as plt

# Define parameters
L = 2.0  # length of the domain
T = 0.625  # time of the simulation
nx = 41  # number of spatial points
nt = 25  # number of time steps
dx = L / (nx - 1)  # spatial increment
dt = T / (nt - 1)  # time increment
cfl = dx / dt  # CFL condition

# Discretize space
x = np.linspace(0, L, nx)

# Set up the initial wave profile
u = np.ones(nx)
u[int(.5 / dx):int(1 / dx + 1)] = 2.0

# Set up the plot
plt.figure()
plt.plot(x, u, marker='o', linestyle='--')
plt.ylim([0.5, 2.5])
plt.xlabel("x")
plt.ylabel("u")
plt.title("Initial condition")
plt.show()

# Iterate using the finite difference scheme
for n in range(nt):
    un = u.copy()
    for i in range(1, nx):
        u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i - 1])

# Plot the wave evolution
plt.figure()
plt.plot(x, u, marker='o', linestyle='--')
plt.ylim([1, 2.5])
plt.xlabel("x")
plt.ylabel("u")
plt.title("Wave evolution")
plt.show()
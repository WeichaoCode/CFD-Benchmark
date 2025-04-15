import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2.0   # length of the domain 
T = 1.0   # time of the simulation 
nx = 41  # number of grid points in space
nt = 21  # number of time steps
c = 1.0  # wave speed

# Discretize space and time
dx = L / (nx - 1)  # space increment
dt = T / (nt - 1)  # time increment
CFL = c * dt / dx  # Courant-Friedrichs-Lewy number

# Check stability
if CFL >= 1.0:
    print("WARNING: The CFL condition is not met.")

# Initial condition
u0 = np.ones(nx)  # array of ones for the initial condition
u0[int(.5 / dx):int(1 / dx + 1)] = 2.0  # set u = 2 between 0.5 and 1

# Initialize solution array
u = u0.copy()

# Solution
for n in range(nt):
    un = u.copy()
    for i in range(1, nx - 1):
        u[i] = un[i] - c * dt / (2 * dx) * (un[i + 1] - un[i - 1])

# Plot
plt.figure(figsize=(6, 4))
plt.plot(np.linspace(0, 2, nx), u0, label='Initial')
plt.plot(np.linspace(0, 2, nx), u, label='Current')
plt.title('1D Linear Convection')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()
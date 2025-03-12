import numpy as np
import matplotlib.pyplot as plt

# Step 1: define parameters
L = 2 * np.pi  # length of the domain
T = 1.0        # time interval
nx = 100       # number of grid points in space
nt = 1000      # number of time steps
nu = 0.07      # viscosity

dx = L / nx                             # spatial resolution
dt = T / nt                             # time step size
cfl = nu * dt / dx**2                   # CFL condition

x = np.linspace(0, L, nx)               # spatial grid points

# Step 2: Initial condition
def phi(x):
    return np.exp(-x**2 / (4 * nu)) + np.exp(-(x - 2*np.pi)**2 / (4 * nu))

dx_phi = -(0.5 / nu) * (phi(x + dx) - phi(x))

u = -2 * nu * dx_phi / phi(x) + 4       # velocity field

# Step 3: Solve using FDM
for n in range(nt):
    un = u.copy()
    u[1:-1] = (un[1:-1] -
               un[1:-1] * dt / dx * 
               (un[1:-1] - un[:-2]) + 
               nu * dt / dx**2 * 
               (un[2:] - 2 * un[1:-1] + un[:-2]))
    
    # Step 4: Apply periodic boundary conditions
    u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + nu * dt / dx**2 * (un[1] - 2 * un[0] + un[-2])
    u[-1] = un[-1] - un[-1] * dt / dx * (un[-1] - un[-3]) + nu * dt / dx**2 * (un[0] - 2 * un[-1] + un[-2])

# Step 5: plot
plt.figure(figsize=(11, 7), dpi=100)
plt.plot(x, phi(x), marker='o', lw=2, label='Analytical')
plt.plot(x, u, marker='x', lw=2, label='Numerical')
plt.xlim([0, 2 * np.pi])
plt.ylim([0, 10])
plt.legend()
plt.show()
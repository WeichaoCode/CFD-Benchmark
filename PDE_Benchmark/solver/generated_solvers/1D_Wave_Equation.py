import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# Define the MMS solution
def u_exact(x, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.cos(t)

# Define the source term
def f(x, t):
    return np.exp(-t) * (np.sin(np.pi * x) * (1 - 2 * np.pi**2) * np.cos(t) - 2 * np.sin(np.pi * x) * np.sin(t))

# Define the wave speed
c = 1.0

# Define the spatial and temporal grid parameters
L = 1.0  # length of the domain
T = 1.0  # total time
nx = 100  # number of spatial grid points
nt = 100  # number of time steps
dx = L / (nx - 1)  # spatial grid size
dt = T / (nt - 1)  # time step size

# Ensure the CFL condition is satisfied
assert c * dt / dx <= 1.0, "CFL condition is not satisfied"

# Create the spatial and temporal grids
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# Initialize the solution array
u = np.zeros((nt, nx))

# Set the initial condition
u[0, :] = u_exact(x, 0)

# Time-stepping loop
for n in range(nt - 1):
    # Compute the second spatial derivative using the central difference scheme
    d2u_dx2 = np.roll(u[n, :], -1) - 2 * u[n, :] + np.roll(u[n, :], 1)
    d2u_dx2 /= dx**2

    # Compute the new solution
    u[n + 1, :] = 2 * u[n, :] - u[n - 1, :] + c**2 * dt**2 * (d2u_dx2 + f(x, t[n]))

# Compute the exact solution
u_exact_vals = u_exact(x[:, np.newaxis], t[np.newaxis, :])

# Compute the absolute error
error = np.abs(u - u_exact_vals)

# Plot the numerical solution
plt.figure()
plt.imshow(u, extent=[0, L, 0, T], origin='lower', aspect='auto')
plt.colorbar(label='u')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Numerical solution')

# Plot the exact solution
plt.figure()
plt.imshow(u_exact_vals, extent=[0, L, 0, T], origin='lower', aspect='auto')
plt.colorbar(label='u')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Exact solution')

# Plot the absolute error
plt.figure()
plt.imshow(error, extent=[0, L, 0, T], origin='lower', aspect='auto')
plt.colorbar(label='Error')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Absolute error')

plt.show()
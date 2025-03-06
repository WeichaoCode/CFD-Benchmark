import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Define the MMS solution
def u_exact(x, t):
    return np.exp(-t) * np.sin(np.pi * x)

# Define the source term
def f(x, t):
    return np.exp(-t) * (np.pi**2 * np.sin(np.pi * x) - np.pi * np.cos(np.pi * x))

# Define the initial condition
def u_initial(x):
    return u_exact(x, 0)

# Define the boundary conditions
def u_boundary(x, t):
    return u_exact(x, t)

# Define the grid parameters
L = 1.0  # length of the domain
N = 100  # number of grid points
dx = L / (N - 1)  # grid spacing
x = np.linspace(0, L, N)  # grid points
dt = 0.001  # time step size
T = 0.1  # final time
Nt = int(T / dt)  # number of time steps
nu = 0.01  # viscosity

# Initialize the solution array
u = np.empty((Nt, N))

# Set the initial condition
u[0, :] = u_initial(x)

# Set the boundary conditions
u[:, 0] = u_boundary(0, np.arange(Nt) * dt)
u[:, -1] = u_boundary(L, np.arange(Nt) * dt)

# Construct the matrix for the implicit scheme
I = sparse.eye(N - 2)
A = sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(N - 2, N - 2))
A = I + nu * dt / dx**2 * A

# Time-stepping loop
for n in range(Nt - 1):
    B = u[n, 1:-1] - dt * u[n, 1:-1] * (u[n, 2:] - u[n, :-2]) / (2 * dx) + dt * f(x[1:-1], n * dt)
    u[n + 1, 1:-1] = spsolve(A, B)

# Compute the exact solution
u_exact = u_exact(x[:, None], np.arange(Nt)[None, :] * dt)

# Compute the absolute error
error = np.abs(u - u_exact)

# Plot the numerical solution, the exact solution, and the error
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(u, extent=[0, T, 0, L], origin='lower', aspect='auto')
plt.colorbar()
plt.title('Numerical solution')

plt.subplot(132)
plt.imshow(u_exact, extent=[0, T, 0, L], origin='lower', aspect='auto')
plt.colorbar()
plt.title('Exact solution')

plt.subplot(133)
plt.imshow(error, extent=[0, T, 0, L], origin='lower', aspect='auto')
plt.colorbar()
plt.title('Absolute error')

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

# Define the MMS solution and its derivatives
def u_exact(x, t):
    return np.exp(-t) * np.sin(np.pi * x)

def f_source(x, t):
    return np.exp(-t) * (np.pi**2 - 1) * np.sin(np.pi * x)

# Define the grid parameters
L = 1.0  # length of the domain
T = 1.0  # time of the simulation
nx = 100  # number of grid points in x
nt = 100  # number of time steps
dx = L / (nx - 1)  # grid size in x
dt = T / (nt - 1)  # time step size
nu = 0.1  # diffusion coefficient

# Create the grid
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

# Initialize the solution array
u = np.zeros((nt, nx))

# Set the initial condition
u[0, :] = u_exact(x, 0)

# Create the finite difference matrix
diagonals = [-nu * dt / dx**2, 1 + 2 * nu * dt / dx**2, -nu * dt / dx**2]
offsets = [-1, 0, 1]
A = sparse.diags(diagonals, offsets, shape=(nx, nx)).tocsc()

# Time-stepping loop
for n in range(nt - 1):
    # Compute the source term
    f = f_source(x, t[n])
    
    # Apply the finite difference scheme
    u[n+1, :] = A.dot(u[n, :]) + dt * f

    # Apply the boundary conditions
    u[n+1, 0] = u_exact(0, t[n+1])
    u[n+1, -1] = u_exact(L, t[n+1])

# Compute the exact solution
u_exact_sol = u_exact(x[:, np.newaxis], t[np.newaxis, :])

# Compute the absolute error
error = np.abs(u - u_exact_sol)

# Plot the numerical solution, exact solution, and error
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.contourf(t, x, u.T)
plt.title('Numerical Solution')
plt.xlabel('Time')
plt.ylabel('Space')

plt.subplot(132)
plt.contourf(t, x, u_exact_sol.T)
plt.title('Exact Solution')
plt.xlabel('Time')
plt.ylabel('Space')

plt.subplot(133)
plt.contourf(t, x, error.T)
plt.title('Absolute Error')
plt.xlabel('Time')
plt.ylabel('Space')

plt.tight_layout()
plt.show()
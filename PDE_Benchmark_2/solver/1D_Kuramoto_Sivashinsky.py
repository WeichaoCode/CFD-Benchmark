import numpy as np
import matplotlib.pyplot as plt

# Computational domain
L = 1.0
T = 1.0

# Number of grid points
nx = 100
nt = 1000

# Grid spacing and time step
dx = L / (nx - 1)
dt = T / (nt - 1)

# Diffusion coefficients
nu2 = 1.0
nu4 = 1.0

# Manufactured solution
def u_exact(x, t):
    return np.exp(-t) * np.sin(np.pi*x)

# Source term
def f(x, t):
    return (np.pi**2-1)*np.exp(-t)*np.sin(np.pi*x) - 2*np.pi*np.exp(-t)*np.cos(np.pi*x)**2

# Initialize solution vectors
u = np.zeros(nx)
u_new = np.zeros(nx)

# Spatial grid
x_grid = np.linspace(0, L, nx)

# Initial condition
u[:] = u_exact(x_grid, 0)

# Time-stepping loop
for n in range(nt):
    # Current time
    t = n * dt
    # Second and fourth derivatives (central difference)
    u_xx = np.roll(u, -1) - 2*u + np.roll(u, 1)
    u_xx /= dx**2
    u_xxxx = np.roll(u, -2) - 4*np.roll(u, -1) + 6*u - 4*np.roll(u, 1) + np.roll(u, 2)
    u_xxxx /= dx**4
    # Nonlinear term (first derivative, upwind scheme)
    u_x = np.where(u > 0, u - np.roll(u, 1), np.roll(u, -1) - u)
    u_x /= dx
    # Right-hand side
    rhs = nu2 * u_xx + nu4 * u_xxxx + 0.5 * u_x**2 - f(x_grid, t)
    # Time stepping (forward Euler)
    u_new[:] = u[:] + dt * rhs
    # Update solution vector
    u, u_new = u_new, u

# Compute analytical solution
u_analytical = u_exact(x_grid, T)

# Plot numerical solution, analytical solution and error
plt.figure()
plt.plot(x_grid, u, label='Numerical')
plt.plot(x_grid, u_analytical, label='Analytical')
plt.legend()

plt.figure()
plt.plot(x_grid, np.abs(u-u_analytical), label='Error')
plt.legend()

plt.show()
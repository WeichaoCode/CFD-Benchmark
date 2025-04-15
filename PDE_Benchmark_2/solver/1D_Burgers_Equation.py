# Import the libraries
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
nu = 0.1 # viscosity
nx = 81  # grid points
dt = 0.025 # time step
nt = 100 # time levels
L = 2.0 # length of domain
dx = L / (nx - 1) # grid spacing

# Create grid
x = np.linspace(0.0, L, num=nx)

# Define initial conditions
u0 = np.zeros(nx)
mask = np.where(np.logical_and(x >= 0.5, x <= 1.0))
u0[mask] = 2.0

# Define analytical(mms) solution
def compute_exact_solution(t):
    return np.exp(-t) * np.sin(np.pi * x)

# Define f(x, t) according to the MMS
def rhs_term(u, t):
    f = np.exp(-t) * np.pi * np.cos(np.pi * x) - 2 * nu * np.exp(-t) * np.pi**2 * np.sin(np.pi * x)
    return f + u * f

# Define the conservation of mass function
def convection_diffusion(u, dt, dx, nu):
    un = u.copy()
    u[1:-1] = (un[1:-1] -
               un[1:-1] * dt / dx * (un[1:-1] - un[:-2]) +
               nu * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[:-2]))
    return u

# Time marching loop
u = u0.copy()
u_analytical = compute_exact_solution(0)
for n in range(nt):
    un = u.copy()
    u = convection_diffusion(un, dt, dx, nu)
    u_analytical = compute_exact_solution(n * dt)

# Plot the final solution along with the exact solution
plt.figure(figsize=(6.0, 4.0))
plt.xlabel('x')
plt.ylabel('u')
plt.grid()
plt.plot(x, u, label='Numerical',
             color='C0', linestyle='-', linewidth=2)
plt.plot(x, u_analytical, label='Analytical',
             color='C1', linestyle='--', linewidth=2)
plt.legend()
plt.xlim(0.0, L)
plt.ylim(0.0, 2.5);
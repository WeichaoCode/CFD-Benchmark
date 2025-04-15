import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# Define the MMS solution
def u_exact(x, t):
    return np.exp(-t) * np.sin(np.pi * x)

# Define the source term
def f(x, t):
    return np.exp(-t) * (np.pi * np.cos(np.pi * x) - np.sin(np.pi * x))

# Define the convection speed
c = 1.0

# Define the spatial domain
x_start = 0.0
x_end = 1.0

# Define the temporal domain
t_start = 0.0
t_end = 1.0

# Define the grid resolution
nx = 100
nt = 100

# Define the grid spacing
dx = (x_end - x_start) / (nx - 1)
dt = (t_end - t_start) / (nt - 1)

# Define the CFL condition
CFL = c * dt / dx

# Ensure the CFL condition is satisfied
assert CFL <= 1, "CFL condition is not satisfied"

# Initialize the grid
x = np.linspace(x_start, x_end, nx)
t = np.linspace(t_start, t_end, nt)

# Initialize the solution array
u = np.zeros((nt, nx))

# Apply the initial condition
u[0, :] = u_exact(x, t_start)

# Apply the boundary condition
u[:, 0] = u_exact(x_start, t)

# Time marching
for n in range(nt - 1):
    # Spatial marching
    for i in range(1, nx):
        # Upwind finite difference scheme
        u[n + 1, i] = u[n, i] - CFL * (u[n, i] - u[n, i - 1]) + dt * f(x[i], t[n])

# Compute the MMS solution
u_mms = u_exact(x, np.linspace(t_start, t_end, nt))

# Compute the absolute error
error = np.abs(u - u_mms)

# Plot the numerical solution
plt.figure()
plt.contourf(x, t, u, cmap='viridis')
plt.title('Numerical Solution')
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()
plt.show()

# Plot the MMS solution
plt.figure()
plt.contourf(x, t, u_mms, cmap='viridis')
plt.title('MMS Solution')
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()
plt.show()

# Plot the absolute error
plt.figure()
plt.contourf(x, t, error, cmap='viridis')
plt.title('Absolute Error')
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()
plt.show()
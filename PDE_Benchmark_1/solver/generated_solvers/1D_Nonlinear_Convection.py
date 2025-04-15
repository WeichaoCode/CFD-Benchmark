import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# Define the MMS solution
def u_exact(x, t):
    return np.exp(-t) * np.sin(np.pi * x)

# Define the source term
def f(x, t):
    return -np.exp(-t) * np.sin(np.pi * x) + np.pi * np.exp(-t) * np.cos(np.pi * x) - np.pi**2 * np.exp(-t) * np.sin(np.pi * x)

# Define the initial condition
def u_initial(x):
    return u_exact(x, 0)

# Define the boundary conditions
def u_boundary(t):
    return u_exact(0, t), u_exact(1, t)

# Define the grid resolution and time step size
nx = 100
nt = 100
dx = 1.0 / (nx - 1)
dt = 1.0 / (nt - 1)
x = np.linspace(0, 1, nx)
t = np.linspace(0, 1, nt)

# Initialize the solution array
u = np.zeros((nt, nx))

# Apply the initial condition
u[0, :] = u_initial(x)

# Apply the boundary conditions
u[:, 0], u[:, -1] = u_boundary(t)

# Solve the PDE using a finite difference method
for n in range(nt - 1):
    for i in range(1, nx - 1):
        u[n+1, i] = u[n, i] - dt/dx * u[n, i] * (u[n, i] - u[n, i-1]) + dt * f(x[i], t[n])

# Compute the MMS solution
u_mms = u_exact(x, t[:, np.newaxis])

# Compute the absolute error
error = np.abs(u - u_mms)

# Plot the numerical solution
plt.figure()
plt.contourf(x, t, u, cmap='viridis')
plt.title('Numerical Solution')
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()

# Plot the MMS solution
plt.figure()
plt.contourf(x, t, u_mms, cmap='viridis')
plt.title('MMS Solution')
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()

# Plot the absolute error
plt.figure()
plt.contourf(x, t, error, cmap='viridis')
plt.title('Absolute Error')
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()

plt.show()
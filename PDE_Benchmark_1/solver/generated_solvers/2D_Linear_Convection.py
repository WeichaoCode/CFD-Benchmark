import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# Define the MMS solution
def u_exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)

# Define the source term
def f(x, y, t):
    return np.exp(-t) * ((1 + np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y) - np.pi * x * np.cos(np.pi * x) * np.sin(np.pi * y) - np.pi * y * np.sin(np.pi * x) * np.cos(np.pi * y))

# Define the convection coefficients
c_x = 1.0
c_y = 1.0

# Define the grid resolution and time step size
nx, ny = 100, 100
dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
dt = 0.001
nt = 100

# Initialize the grid
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# Initialize the solution and the exact solution
u = u_exact(X, Y, 0)
u_exact_sol = u_exact(X, Y, nt * dt)

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] - c_x * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) - c_y * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) + dt * f(X[1:-1, 1:-1], Y[1:-1, 1:-1], (n+1) * dt))

# Compute the absolute error
error = np.abs(u - u_exact_sol)

# Plot the numerical solution, the exact solution, and the error
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.contourf(X, Y, u, cmap='viridis')
plt.title('Numerical Solution')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.contourf(X, Y, u_exact_sol, cmap='viridis')
plt.title('Exact Solution')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.contourf(X, Y, error, cmap='viridis')
plt.title('Absolute Error')
plt.colorbar()

plt.tight_layout()
plt.show()
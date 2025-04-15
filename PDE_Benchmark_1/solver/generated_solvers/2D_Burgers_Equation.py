import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

# Define the MMS solution
def u_exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi*x) * np.sin(np.pi*y)

def v_exact(x, y, t):
    return np.exp(-t) * np.cos(np.pi*x) * np.cos(np.pi*y)

# Define the source terms
def f_u(x, y, t):
    return -u_exact(x, y, t) + np.pi*np.exp(-t)*np.cos(np.pi*x)*np.sin(np.pi*y) - np.pi**2*u_exact(x, y, t)

def f_v(x, y, t):
    return -v_exact(x, y, t) - np.pi*np.exp(-t)*np.sin(np.pi*x)*np.cos(np.pi*y) - np.pi**2*v_exact(x, y, t)

# Define the grid
N = 100
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

# Define the time step and viscosity
dt = 0.01
nu = 0.1

# Initialize the solution
u = u_exact(X, Y, 0)
v = v_exact(X, Y, 0)

# Time-stepping loop
for t in np.arange(0, 1, dt):
    # Compute the source terms
    Fu = f_u(X, Y, t)
    Fv = f_v(X, Y, t)

    # Compute the derivatives using FFT
    u_x = fftpack.diff(u, period=1.0, axis=1)
    u_y = fftpack.diff(u, period=1.0, axis=0)
    v_x = fftpack.diff(v, period=1.0, axis=1)
    v_y = fftpack.diff(v, period=1.0, axis=0)

    # Update the solution
    u_new = u - dt*(u*u_x + v*u_y - nu*(u_x**2 + u_y**2) - Fu)
    v_new = v - dt*(u*v_x + v*v_y - nu*(v_x**2 + v_y**2) - Fv)

    # Update the solution
    u = u_new
    v = v_new

# Compute the exact solution
u_exact = u_exact(X, Y, 1)
v_exact = v_exact(X, Y, 1)

# Compute the error
error_u = np.abs(u - u_exact)
error_v = np.abs(v - v_exact)

# Plot the numerical solution
plt.figure()
plt.contourf(X, Y, u)
plt.colorbar()
plt.title('Numerical solution for u')
plt.show()

# Plot the exact solution
plt.figure()
plt.contourf(X, Y, u_exact)
plt.colorbar()
plt.title('Exact solution for u')
plt.show()

# Plot the error
plt.figure()
plt.contourf(X, Y, error_u)
plt.colorbar()
plt.title('Error for u')
plt.show()
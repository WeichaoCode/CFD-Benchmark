import numpy as np
import matplotlib.pyplot as plt

# Define grid parameters
nx = 101  # number of grid points in x
nt = 500  # number of time steps
c = 1.0   # wave speed
dt = 0.01 # time step size
dx = 2.0 / (nx - 1)  # spatial grid size

# Define source function f(x, t)
def f(x, t):
    return np.exp(-t) * np.pi**2 * np.sin(np.pi*x) * (np.cos(t) - 1)

# Initialize solution arrays
u = np.zeros(nx)
u_new = np.zeros(nx)
u_old = np.zeros(nx)

# Define x points
x = np.linspace(0, 2, nx)

# Initial conditions (from Manufactured Solution)
u[:] = np.sin(np.pi*x)
u_old = u.copy()

# Time steps
for n in range(1, nt):
    u_new = 2*u - u_old + (c*dt/dx)**2 * (np.roll(u, -1) - 2*u + np.roll(u, 1)) + dt**2 * f(x, n*dt)
    
    # Update solution arrays
    u_old = u.copy()
    u = u_new.copy()

# Manufactured Solution for comparison
u_mms = np.exp(-nt*dt) * np.sin(np.pi*x) * np.cos(nt*dt)

# Calculate absolute error
error = np.abs(u - u_mms)

# Plot numerical solution, exact solution, and error
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(x, u, label='Numerical')
plt.plot(x, u_mms, label='Exact')
plt.title('Solution at nt = {}'.format(nt))
plt.xlabel('x')
plt.ylabel('u')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(x, u_mms, label='Exact')
plt.title('Exact Solution')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(x, error)
plt.title('Absolute Error')
plt.xlabel('x')
plt.ylabel('Error')

plt.tight_layout()
plt.show()
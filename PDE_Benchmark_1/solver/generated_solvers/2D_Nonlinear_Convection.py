import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# Define the MMS solution
def u_exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi*x) * np.sin(np.pi*y)

def v_exact(x, y, t):
    return np.exp(-t) * np.cos(np.pi*x) * np.cos(np.pi*y)

# Define the source terms
def f(x, y, t):
    return -u_exact(x, y, t) + np.pi*np.exp(-t)*np.sin(np.pi*x)*np.sin(np.pi*y)*(np.cos(np.pi*x) + np.cos(np.pi*y))

def f_v(x, y, t):
    return -v_exact(x, y, t) - np.pi*np.exp(-t)*np.cos(np.pi*x)*np.cos(np.pi*y)*(np.sin(np.pi*x) + np.sin(np.pi*y))

# Define the grid resolution and time step size
nx, ny, nt = 101, 101, 100
dx, dy, dt = 1.0/(nx-1), 1.0/(ny-1), 0.01

# Create the grid
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# Initialize the solution arrays
u = u_exact(X, Y, 0)
v = v_exact(X, Y, 0)

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Compute the next time step
    u[1:-1, 1:-1] = un[1:-1, 1:-1] - dt*(un[1:-1, 1:-1]*(un[1:-1, 1:-1] - un[:-2, 1:-1])/dx + vn[1:-1, 1:-1]*(un[1:-1, 1:-1] - un[1:-1, :-2])/dy) + dt*f(X[1:-1, 1:-1], Y[1:-1, 1:-1], n*dt)
    v[1:-1, 1:-1] = vn[1:-1, 1:-1] - dt*(un[1:-1, 1:-1]*(vn[1:-1, 1:-1] - vn[:-2, 1:-1])/dx + vn[1:-1, 1:-1]*(vn[1:-1, 1:-1] - vn[1:-1, :-2])/dy) + dt*f_v(X[1:-1, 1:-1], Y[1:-1, 1:-1], n*dt)
    
    # Apply the boundary conditions
    u[0, :], u[-1, :], u[:, 0], u[:, -1] = u_exact(x, 0, (n+1)*dt), u_exact(x, 1, (n+1)*dt), u_exact(0, y, (n+1)*dt), u_exact(1, y, (n+1)*dt)
    v[0, :], v[-1, :], v[:, 0], v[:, -1] = v_exact(x, 0, (n+1)*dt), v_exact(x, 1, (n+1)*dt), v_exact(0, y, (n+1)*dt), v_exact(1, y, (n+1)*dt)

# Compute the MMS solution and the absolute error
u_mms = u_exact(X, Y, nt*dt)
v_mms = v_exact(X, Y, nt*dt)
error_u = np.abs(u - u_mms)
error_v = np.abs(v - v_mms)

# Plot the numerical solution, the MMS solution, and the absolute error
plt.figure(figsize=(15, 10))

plt.subplot(231)
plt.contourf(X, Y, u, cmap='viridis')
plt.title('Numerical Solution u')
plt.colorbar()

plt.subplot(232)
plt.contourf(X, Y, u_mms, cmap='viridis')
plt.title('MMS Solution u')
plt.colorbar()

plt.subplot(233)
plt.contourf(X, Y, error_u, cmap='viridis')
plt.title('Absolute Error u')
plt.colorbar()

plt.subplot(234)
plt.contourf(X, Y, v, cmap='viridis')
plt.title('Numerical Solution v')
plt.colorbar()

plt.subplot(235)
plt.contourf(X, Y, v_mms, cmap='viridis')
plt.title('MMS Solution v')
plt.colorbar()

plt.subplot(236)
plt.contourf(X, Y, error_v, cmap='viridis')
plt.title('Absolute Error v')
plt.colorbar()

plt.tight_layout()
plt.show()
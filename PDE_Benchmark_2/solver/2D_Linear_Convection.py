import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to compute u based on Manufactured Solution (MMS)
def mms_solution(x, y, t):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y)

# Function to compute the source term f
def source_term(x, y, t):
    return -np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y) + \
           (c_x*np.pi*np.exp(-t)*np.cos(np.pi * x)*np.sin(np.pi * y) + 
           c_y*np.pi*np.exp(-t)*np.sin(np.pi * x)*np.cos(np.pi * y))

# Simulation parameters
nx = 50
ny = 50
nt = 100
c_x = 1
c_y = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = .001

# Initialize grid
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x,y)

# Initialize u and un
u = np.empty((ny, nx))
un = np.empty((ny, nx))

# Set initial condition
u[:] = mms_solution(X, Y, 0)

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    f = source_term(X, Y, n*dt)

    # Finite-Difference with first-order upwind scheme (explicit)
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] - 
                     (c_x * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2])) -
                     (c_y * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1])) +
                     dt * f[1:-1, 1:-1])
        
    # Enforce boundary conditions (using MMS)
    u[0, :] = mms_solution(X[0, :], Y[0, :], n * dt)
    u[-1, :] = mms_solution(X[-1, :], Y[-1, :], n * dt)
    u[:, 0] = mms_solution(X[:, 0], Y[:, 0], n * dt)
    u[:, -1] = mms_solution(X[:, -1], Y[:, -1], n * dt)

# Compute MMS (exact solution) and absolute error
u_exact = mms_solution(X, Y, nt * dt)
error = np.abs(u - u_exact)

# Plot solution and error
fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

ax.plot_surface(X, Y, u, cmap='viridis')
ax.set_title('Numerical Solution')
plt.show()

ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, u_exact, cmap='viridis')
ax.set_title('Exact Solution (MMS)')
plt.show()

ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, error, cmap='plasma')
ax.set_title('Absolute Error')
plt.show()
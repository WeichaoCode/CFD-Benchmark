import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the grid
nx, ny, nt = 101, 101, 101  # number of points in x, y and t
dx, dy, dt = 1.0/(nx-1), 1.0/(ny-1), 0.01  # grid sizes
x = np.linspace(0, 1, nx)  # x grid
y = np.linspace(0, 1, ny)  # y grid
t = np.linspace(0, 1, nt)  # t grid
X, Y = np.meshgrid(x, y)  # create meshgrid

# Define the diffusion coefficient
alpha = 0.1

# Define the Manufactured Solution (MMS)
def u_exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi*x) * np.sin(np.pi*y)

# Define the source term
def f(x, y, t):
    return np.exp(-t) * ((1 + 2*np.pi**2)*np.sin(np.pi*x)*np.sin(np.pi*y) - np.pi*x*np.cos(np.pi*x)*np.sin(np.pi*y) - np.pi*y*np.sin(np.pi*x)*np.cos(np.pi*y))

# Initialize the solution array
u = np.zeros((nx, ny, nt))

# Set initial condition
u[:,:,0] = u_exact(X, Y, 0)

# Time-stepping loop
for n in range(nt-1):
    # Compute the next time step
    u[1:-1, 1:-1, n+1] = u[1:-1, 1:-1, n] + alpha*dt*(np.diff(u[:-2, 1:-1, n],2,axis=0)/dx**2 + np.diff(u[1:-1, :-2, n],2,axis=1)/dy**2) + dt*f(X[1:-1, 1:-1], Y[1:-1, 1:-1], t[n])
    # Apply boundary conditions
    u[:, 0, n+1] = u_exact(x, 0, t[n+1])
    u[:, -1, n+1] = u_exact(x, 1, t[n+1])
    u[0, :, n+1] = u_exact(0, y, t[n+1])
    u[-1, :, n+1] = u_exact(1, y, t[n+1])

# Compute the exact solution
u_exact_sol = u_exact(X, Y, t[-1])

# Compute the absolute error
error = np.abs(u_exact_sol - u[:,:,-1])

# Plot the numerical solution
fig = plt.figure(figsize=(11,7), dpi=100)
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, u[:,:,-1], cmap='viridis')
plt.title('Numerical Solution')
plt.show()

# Plot the exact solution
fig = plt.figure(figsize=(11,7), dpi=100)
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, u_exact_sol, cmap='viridis')
plt.title('Exact Solution')
plt.show()

# Plot the absolute error
fig = plt.figure(figsize=(11,7), dpi=100)
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, error, cmap='viridis')
plt.title('Absolute Error')
plt.show()
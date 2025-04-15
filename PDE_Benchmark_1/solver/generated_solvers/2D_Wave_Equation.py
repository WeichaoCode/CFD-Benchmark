import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the grid
nx, ny, nt = 101, 101, 101  # number of points in x, y and t
x = np.linspace(0, 1, nx)  # x grid
y = np.linspace(0, 1, ny)  # y grid
t = np.linspace(0, 1, nt)  # time grid
dx, dy, dt = x[1]-x[0], y[1]-y[0], t[1]-t[0]  # grid spacings
c = 1.0  # wave speed

# Define the MMS solution and its second derivatives
u_exact = lambda x, y, t: np.exp(-t) * np.sin(np.pi*x) * np.sin(np.pi*y) * np.cos(t)
uxx = lambda x, y, t: -np.pi**2 * u_exact(x, y, t)
uyy = lambda x, y, t: -np.pi**2 * u_exact(x, y, t)
utt = lambda x, y, t: -u_exact(x, y, t)

# Define the source term
f = lambda x, y, t: utt(x, y, t) - c**2 * (uxx(x, y, t) + uyy(x, y, t))

# Initialize the solution arrays
u = np.zeros((nx, ny, nt))  # numerical solution
ue = np.zeros((nx, ny, nt))  # exact solution

# Set initial conditions
u[:,:,0] = u_exact(x[:,None], y[None,:], t[0])
ue[:,:,0] = u_exact(x[:,None], y[None,:], t[0])

# Time-stepping loop
for k in range(1, nt):
    # Compute the numerical solution
    u[1:-1,1:-1,k] = 2*u[1:-1,1:-1,k-1] - u[1:-1,1:-1,k-2] + c**2 * dt**2 * (
        (u[:-2,1:-1,k-1] - 2*u[1:-1,1:-1,k-1] + u[2:,1:-1,k-1]) / dx**2 +
        (u[1:-1,:-2,k-1] - 2*u[1:-1,1:-1,k-1] + u[1:-1,2:,k-1]) / dy**2 +
        f(x[1:-1,None], y[None,1:-1], t[k-1])
    )
    # Compute the exact solution
    ue[:,:,k] = u_exact(x[:,None], y[None,:], t[k])

# Compute the absolute error
error = np.abs(u - ue)

# Plot the numerical solution, exact solution and error
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(x, y, u[:,:,-1])
ax1.set_title('Numerical Solution')
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(x, y, ue[:,:,-1])
ax2.set_title('Exact Solution')
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(x, y, error[:,:,-1])
ax3.set_title('Absolute Error')
plt.show()
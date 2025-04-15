import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
nu = 0.01  # Diffusion coefficient
nx, ny = 101, 101  # Grid points
Lx, Ly = 2.0, 2.0  # Spatial domain [0,2] × [0,2]
dx = Lx / (nx - 1)  # Spatial step in x
dy = Ly / (ny - 1)  # Spatial step in y
dt = 0.001  # Time step size
nt = 200  # Number of time steps

# Grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize velocity components u and v
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial condition: Hat function
u[(Y >= 0.5) & (Y <= 1.0) & (X >= 0.5) & (X <= 1.0)] = 2
v[(Y >= 0.5) & (Y <= 1.0) & (X >= 0.5) & (X <= 1.0)] = 2

# Time-stepping loop
for _ in range(nt):
    un = u.copy()
    vn = v.copy()

    # Update u using Burgers' equation
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     dt / dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     dt / dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1]) +
                     nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                     nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]))

    # Update v using Burgers' equation
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     dt / dx * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     dt / dy * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) +
                     nu * dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                     nu * dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1]))

    # Apply boundary conditions: u = 1, v = 1 for x=0,2 and y=0,2
    u[0, :], u[-1, :], u[:, 0], u[:, -1] = 1, 1, 1, 1
    v[0, :], v[-1, :], v[:, 0], v[:, -1] = 1, 1, 1, 1

# Plot results: 3D Surface Plots
fig = plt.figure(figsize=(12, 5))

# 3D plot for u
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, u, cmap='jet', edgecolor='k')
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("u(x, y)")
ax1.set_title("2D Burgers' Equation - u Velocity Field")

# 3D plot for v
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, v, cmap='jet', edgecolor='k')
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("v(x, y)")
ax2.set_title("2D Burgers' Equation - v Velocity Field")

np.save("u_pred.npy", u)
np.save("v_pred.npy", v)

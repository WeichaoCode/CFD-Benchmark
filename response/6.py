import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
nx, ny = 101, 101  # Grid points in x and y directions
Lx, Ly = 2.0, 2.0  # Spatial domain [0,2] × [0,2]
dx = Lx / (nx - 1)  # Spatial step in x
dy = Ly / (ny - 1)  # Spatial step in y
dt = 0.2 * min(dx, dy)  # CFL condition for stability
nt = 100  # Number of time steps

# Grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize velocity fields u and v
u = np.ones((ny, nx))  # Default value everywhere
v = np.ones((ny, nx))  # Default value everywhere

# Initial conditions: u, v = 2 in the central region (0.5 ≤ x, y ≤ 1)
u[(Y >= 0.5) & (Y <= 1.0) & (X >= 0.5) & (X <= 1.0)] = 2
v[(Y >= 0.5) & (Y <= 1.0) & (X >= 0.5) & (X <= 1.0)] = 2

# Time-stepping loop
for _ in range(nt):
    u_new = np.copy(u)
    v_new = np.copy(v)

    # Apply backward differences for spatial derivatives
    u_new[1:, 1:] = (u[1:, 1:] -
                     (dt / dx) * u[1:, 1:] * (u[1:, 1:] - u[1:, :-1]) -
                     (dt / dy) * v[1:, 1:] * (u[1:, 1:] - u[:-1, 1:]))

    v_new[1:, 1:] = (v[1:, 1:] -
                     (dt / dx) * u[1:, 1:] * (v[1:, 1:] - v[1:, :-1]) -
                     (dt / dy) * v[1:, 1:] * (v[1:, 1:] - v[:-1, 1:]))

    # Apply boundary conditions: u = 1, v = 1 for x = 0,2 and y = 0,2
    u_new[0, :], u_new[-1, :], u_new[:, 0], u_new[:, -1] = 1, 1, 1, 1
    v_new[0, :], v_new[-1, :], v_new[:, 0], v_new[:, -1] = 1, 1, 1, 1

    # Update solution
    u, v = u_new, v_new

# Plot results: 3D Surface Plots
fig = plt.figure(figsize=(12, 5))

# 3D plot for u
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, u, cmap='jet', edgecolor='k')
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("u(x, y)")
ax1.set_title("2D Convection - u Velocity Field")

# 3D plot for v
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, v, cmap='jet', edgecolor='k')
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("v(x, y)")
ax2.set_title("2D Convection - v Velocity Field")

np.save("u_pred.npy", u)
np.save("v_pred.npy", v)

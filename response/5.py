import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
c = 1.0  # Convection speed
nx, ny = 101, 101  # Grid points in x and y directions
Lx, Ly = 2.0, 2.0  # Domain size
dx = Lx / (nx - 1)  # Spatial step in x
dy = Ly / (ny - 1)  # Spatial step in y
dt = 0.2 * min(dx, dy) / c  # CFL condition for stability
nt = 100  # Number of time steps

# Grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initial condition
u = np.ones((ny, nx))  # Default value everywhere
u[(Y >= 0.5) & (Y <= 1.0) & (X >= 0.5) & (X <= 1.0)] = 2  # Set u = 2 in the given region

# Time-stepping loop
for _ in range(nt):
    u_new = np.copy(u)

    # Apply backward differences for spatial derivatives
    u_new[1:, 1:] = (u[1:, 1:] -
                     (dt / dx) * c * (u[1:, 1:] - u[1:, :-1]) -  # Backward difference in x
                     (dt / dy) * c * (u[1:, 1:] - u[:-1, 1:]))   # Backward difference in y

    # Apply boundary conditions: u = 1 at x = 0,2 and y = 0,2
    u_new[0, :] = 1  # y = 0
    u_new[-1, :] = 1  # y = 2
    u_new[:, 0] = 1  # x = 0
    u_new[:, -1] = 1  # x = 2

    u = u_new  # Update solution

# 3D Surface Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, u, cmap='jet', edgecolor='k')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x, y)")
ax.set_title("2D Linear Convection - 3D Surface Plot")
# plt.show()

np.save("u_pred.npy", u)
np.save("v_pred.npy", u)

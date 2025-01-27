import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
nu = 0.05  # Diffusion coefficient
nx, ny = 101, 101  # Grid points in x and y directions
Lx, Ly = 2.0, 2.0  # Spatial domain [0,2] × [0,2]
dx = Lx / (nx - 1)  # Spatial step in x
dy = Ly / (ny - 1)  # Spatial step in y
dt = 0.25 * min(dx, dy)**2 / nu  # Stability condition for diffusion (CFL condition)
nt = 200  # Number of time steps

# Grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize u
u = np.ones((ny, nx))  # Default value everywhere

# Initial condition: Hat function
u[(Y >= 0.5) & (Y <= 1.0) & (X >= 0.5) & (X <= 1.0)] = 2

# Time-stepping loop
for _ in range(nt):
    u_new = np.copy(u)

    # Apply central differences for spatial derivatives (Explicit scheme)
    u_new[1:-1, 1:-1] = (u[1:-1, 1:-1] +
                         nu * dt * ((u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
                                    (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2))

    # Apply boundary conditions: u = 1 at x = 0,2 and y = 0,2
    u_new[0, :], u_new[-1, :], u_new[:, 0], u_new[:, -1] = 1, 1, 1, 1

    # Update solution
    u = u_new

# Plot the final solution: 3D Surface Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, u, cmap='jet', edgecolor='k')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x, y)")
ax.set_title("2D Diffusion Equation - 3D Surface Plot")
# plt.show()

np.save("u_pred.npy", u)
np.save("v_pred.npy", u)

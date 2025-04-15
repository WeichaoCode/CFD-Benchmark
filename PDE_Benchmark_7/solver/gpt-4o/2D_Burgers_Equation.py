import numpy as np
import matplotlib.pyplot as plt

# Define parameters
nx, ny = 41, 41
nt = 120
sigma = 0.0009
nu = 0.01
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = sigma * dx * dy / nu

# Initialize variables
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial condition: set u, v = 2 in the region 0.5 <= x, y <= 1
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2
v[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time integration loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Compute temporary arrays using central differences
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     dt / dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     dt / dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1]) +
                     nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                     nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]))

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     dt / dx * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     dt / dy * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) +
                     nu * dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                     nu * dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1]))

    # Apply boundary conditions
    u[0, :], u[-1, :], u[:, 0], u[:, -1] = 1, 1, 1, 1
    v[0, :], v[-1, :], v[:, 0], v[:, -1] = 1, 1, 1, 1

# Save the final velocity fields
np.save('/PDE_Benchmark_7/results/prediction/u_2D_Burgers_Equation.npy', u)
np.save('/PDE_Benchmark_7/results/prediction/v_2D_Burgers_Equation.npy', v)

# Visualization
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, u, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('U')
ax.set_title('Velocity field U at final time step')
plt.show()

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, v, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('V')
ax.set_title('Velocity field V at final time step')
plt.show()
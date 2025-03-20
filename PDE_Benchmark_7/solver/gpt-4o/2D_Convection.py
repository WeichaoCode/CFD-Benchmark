import numpy as np
import matplotlib.pyplot as plt

# Define parameters
nx, ny = 101, 101      # Number of grid points
dx = dy = 2 / (nx - 1) # Grid spacing
sigma = 0.2            # CFL number
dt = sigma * dx        # Time step size
nt = 80                # Number of time steps

# Initialize variables
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Set initial conditions: u, v = 2 in the region 0.5 <= x <= 1, 0.5 <= y <= 1
u[int(0.5/dy):int(1/dy + 1), int(0.5/dx):int(1/dx + 1)] = 2
v[int(0.5/dy):int(1/dy + 1), int(0.5/dx):int(1/dx + 1)] = 2

# Time integration loop
for n in range(nt + 1):
    un = u.copy()
    vn = v.copy()

    # Update u and v using the Explicit Euler method with upwind scheme
    u[1:, 1:] = (un[1:, 1:] - 
                 dt * (un[1:, 1:] * (un[1:, 1:] - un[:-1, 1:]) / dx + 
                       vn[1:, 1:] * (un[1:, 1:] - un[1:, :-1]) / dy))
    v[1:, 1:] = (vn[1:, 1:] - 
                 dt * (un[1:, 1:] * (vn[1:, 1:] - vn[:-1, 1:]) / dx + 
                       vn[1:, 1:] * (vn[1:, 1:] - vn[1:, :-1]) / dy))

    # Apply boundary conditions: u, v = 1 on all boundaries
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 1
    v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 1

# Save final velocity fields
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/u_2D_Convection.npy', u)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_7/solver/gpt-4o/v_2D_Convection.npy', v)

# Visualization
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(12, 6))

# Plot u at final time step
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, Y, u, cmap='viridis')
ax.set_title('Velocity field u')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u')

# Plot v at final time step
ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X, Y, v, cmap='viridis')
ax.set_title('Velocity field v')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('v')

plt.tight_layout()
plt.show()
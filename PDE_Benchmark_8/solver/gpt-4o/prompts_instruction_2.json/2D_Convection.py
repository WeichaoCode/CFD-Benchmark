import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
nx, ny = 101, 101
nt = 80
sigma = 0.2
dx = dy = 2 / (nx - 1)
dt = sigma * dx

# Initialize velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Initial condition
u[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2
v[int(0.5 / dy):int(1 / dy + 1), int(0.5 / dx):int(1 / dx + 1)] = 2

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update u and v using upwind scheme
    u[1:, 1:] = (un[1:, 1:] - 
                 dt / dx * un[1:, 1:] * (un[1:, 1:] - un[1:, :-1]) - 
                 dt / dy * vn[1:, 1:] * (un[1:, 1:] - un[:-1, 1:]))
    
    v[1:, 1:] = (vn[1:, 1:] - 
                 dt / dx * un[1:, 1:] * (vn[1:, 1:] - vn[1:, :-1]) - 
                 dt / dy * vn[1:, 1:] * (vn[1:, 1:] - vn[:-1, 1:]))
    
    # Apply Dirichlet boundary conditions
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1
    
    v[0, :] = 1
    v[-1, :] = 1
    v[:, 0] = 1
    v[:, -1] = 1

# Save final solution to .npy files
np.save('u_final.npy', u)
np.save('v_final.npy', v)

# Visualization
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(12, 6))

# Plot u
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, u, cmap='viridis')
ax1.set_title('Velocity field u')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u')

# Plot v
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, v, cmap='viridis')
ax2.set_title('Velocity field v')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('v')

plt.tight_layout()
plt.show()
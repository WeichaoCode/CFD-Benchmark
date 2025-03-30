import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
nx, ny = 41, 41
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
nt = 120
sigma = 0.0009
nu = 0.01
dt = sigma * dx * dy / nu

# Initialize velocity fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Set initial condition: u = v = 2 for 0.5 <= x, y <=1
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)
mask = (X >= 0.5) & (X <=1) & (Y >=0.5) & (Y <=1)
u[mask] = 2
v[mask] = 2

# Time-stepping
for _ in range(nt):
    un = u.copy()
    vn = v.copy()
    
    u[1:-1,1:-1] = (un[1:-1,1:-1] 
                    - un[1:-1,1:-1] * dt / dx * (un[1:-1,1:-1] - un[1:-1,0:-2]) 
                    - vn[1:-1,1:-1] * dt / dy * (un[1:-1,1:-1] - un[0:-2,1:-1])
                    + nu * dt / dx**2 * (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,0:-2])
                    + nu * dt / dy**2 * (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[0:-2,1:-1]))
    
    v[1:-1,1:-1] = (vn[1:-1,1:-1] 
                    - un[1:-1,1:-1] * dt / dx * (vn[1:-1,1:-1] - vn[1:-1,0:-2]) 
                    - vn[1:-1,1:-1] * dt / dy * (vn[1:-1,1:-1] - vn[0:-2,1:-1])
                    + nu * dt / dx**2 * (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,0:-2])
                    + nu * dt / dy**2 * (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[0:-2,1:-1]))
    
    # Apply Dirichlet boundary conditions
    u[:,0] = 1
    u[:,-1] = 1
    u[0,:] = 1
    u[-1,:] = 1
    
    v[:,0] = 1
    v[:,-1] = 1
    v[0,:] = 1
    v[-1,:] = 1

# Plotting u
fig = plt.figure(figsize=(11,7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, u, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u')
plt.show()

# Plotting v
fig = plt.figure(figsize=(11,7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, v, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('v')
plt.show()

# Save the final velocity fields
np.save('u.npy', u)
np.save('v.npy', v)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Grid parameters
Nx = 101
Ny = 101
nt = 500
dx = 2.0/(Nx-1)
dy = 2.0/(Ny-1)
dt = 2.0/nt
nu = 0.01

x = np.linspace(0, 2, Nx)
y = np.linspace(0, 2, Ny)
X, Y = np.meshgrid(x, y)

# Initialize velocity fields
u = np.ones((Ny, Nx))
v = np.ones((Ny, Nx))

# Set initial conditions
u[(Y>=0.5) & (Y<=1.0) & (X>=0.5) & (X<=1.0)] = 2.0
v[(Y>=0.5) & (Y<=1.0) & (X>=0.5) & (X<=1.0)] = 2.0

# Arrays to store solution at next time step
un = np.zeros((Ny, Nx))
vn = np.zeros((Ny, Nx))

def central_diff_x(f):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1))/(2*dx)

def central_diff_y(f):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0))/(2*dy)

def laplacian(f):
    return (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1))/(dx**2) + \
           (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0))/(dy**2)

# Time stepping
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update u
    u[1:-1, 1:-1] = un[1:-1, 1:-1] - \
                    dt * (un[1:-1, 1:-1] * central_diff_x(un)[1:-1, 1:-1] + \
                         vn[1:-1, 1:-1] * central_diff_y(un)[1:-1, 1:-1]) + \
                    nu * dt * laplacian(un)[1:-1, 1:-1]
    
    # Update v
    v[1:-1, 1:-1] = vn[1:-1, 1:-1] - \
                    dt * (un[1:-1, 1:-1] * central_diff_x(vn)[1:-1, 1:-1] + \
                         vn[1:-1, 1:-1] * central_diff_y(vn)[1:-1, 1:-1]) + \
                    nu * dt * laplacian(vn)[1:-1, 1:-1]
    
    # Boundary conditions
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 1
    v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 1

# Plot results
fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, u, cmap=cm.viridis)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u')
ax1.set_title('u velocity')
fig.colorbar(surf1)

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, v, cmap=cm.viridis)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('v')
ax2.set_title('v velocity')
fig.colorbar(surf2)

plt.tight_layout()
plt.show()
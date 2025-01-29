import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
Nx = 101
Ny = 101
nt = 500
dx = 2.0/(Nx-1)
dy = 2.0/(Ny-1)
x = np.linspace(0, 2, Nx)
y = np.linspace(0, 2, Ny)
X, Y = np.meshgrid(x, y)

# Physical parameters
nu = 0.01
dt = 2.0/nt

# Initialize velocity fields
u = np.ones((Ny, Nx))
v = np.ones((Ny, Nx))

# Set initial conditions
u[(Y>=0.5) & (Y<=1.0) & (X>=0.5) & (X<=1.0)] = 2.0
v[(Y>=0.5) & (Y<=1.0) & (X>=0.5) & (X<=1.0)] = 2.0

# Create copies for the next time step
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
    u[1:-1, 1:-1] = un[1:-1, 1:-1] - dt * (
        un[1:-1, 1:-1] * central_diff_x(un)[1:-1, 1:-1] +
        vn[1:-1, 1:-1] * central_diff_y(un)[1:-1, 1:-1]
    ) + nu * dt * laplacian(un)[1:-1, 1:-1]
    
    # Update v
    v[1:-1, 1:-1] = vn[1:-1, 1:-1] - dt * (
        un[1:-1, 1:-1] * central_diff_x(vn)[1:-1, 1:-1] +
        vn[1:-1, 1:-1] * central_diff_y(vn)[1:-1, 1:-1]
    ) + nu * dt * laplacian(vn)[1:-1, 1:-1]
    
    # Apply boundary conditions
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 1
    v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 1

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.contourf(X, Y, u, levels=np.linspace(u.min(), u.max(), 50))
plt.colorbar(label='u velocity')
plt.title('u velocity field')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(122)
plt.contourf(X, Y, v, levels=np.linspace(v.min(), v.max(), 50))
plt.colorbar(label='v velocity')
plt.title('v velocity field')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()
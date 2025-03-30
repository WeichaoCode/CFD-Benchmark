import numpy as np

# Parameters
nu = 1.0
nx = ny = 31
dx = dy = 2.0 / (nx -1)
nt = 50
sigma = 0.25
dt = sigma * dx * dy / nu

# Initialize grid
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# Initialize u
u = np.ones((ny, nx))
u[np.where((X >= 0.5) & (X <=1) & (Y >=0.5) & (Y <=1))] = 2.0

# Time-stepping
for n in range(nt):
    un = u.copy()
    # Compute interior points
    u[1:-1,1:-1] = un[1:-1,1:-1] + nu * dt / dx**2 * (
        un[2:,1:-1] - 2 * un[1:-1,1:-1] + un[0:-2,1:-1]) + \
        nu * dt / dy**2 * (
        un[1:-1,2:] - 2 * un[1:-1,1:-1] + un[1:-1,0:-2])
    # Apply boundary conditions
    u[0,:] = 1
    u[-1,:] = 1
    u[:,0] = 1
    u[:,-1] = 1

# Save the final solution
np.save('u.npy', u)
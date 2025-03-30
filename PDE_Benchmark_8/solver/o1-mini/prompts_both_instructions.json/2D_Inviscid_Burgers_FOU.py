import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = ny = 151
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
nt = 300
sigma = 0.2
dt = sigma * min(dx, dy) / 2

# Initialize fields
u = np.ones((ny, nx))
v = np.ones((ny, nx))

# Apply initial conditions
u_initial_region = np.where((np.linspace(0, 2, nx).reshape(1, nx) >= 0.5) &
                            (np.linspace(0, 2, nx).reshape(1, nx) <= 1.0))
v_initial_region = u_initial_region
u[u_initial_region] = 2
v[v_initial_region] = 2

# Time-stepping
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    # Update interior points using First-Order Upwind scheme
    u[1:,1:] = (un[1:,1:] - 
               un[1:,1:] * dt / dx * (un[1:,1:] - un[1:,:-1]) -
               vn[1:,1:] * dt / dy * (un[1:,1:] - un[:-1,1:]))
    
    v[1:,1:] = (vn[1:,1:] - 
               un[1:,1:] * dt / dx * (vn[1:,1:] - vn[1:,:-1]) -
               vn[1:,1:] * dt / dy * (vn[1:,1:] - vn[:-1,1:]))
    
    # Apply Dirichlet boundary conditions
    u[0,:] = 1
    u[-1,:] = 1
    u[:,0] = 1
    u[:,-1] = 1
    
    v[0,:] = 1
    v[-1,:] = 1
    v[:,0] = 1
    v[:,-1] = 1

# Save final fields
np.save('u.npy', u)
np.save('v.npy', v)

# Visualization
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(8,6))
plt.quiver(X, Y, u, v)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Final Velocity Field')
plt.savefig('velocity_field.png')
plt.show()
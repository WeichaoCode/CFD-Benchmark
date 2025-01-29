import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
Nx = 101
Ny = 101
Nt = 500
dx = 2.0/(Nx-1)
dy = 2.0/(Ny-1)
dt = 2.0/Nt

x = np.linspace(0, 2, Nx)
y = np.linspace(0, 2, Ny)
X, Y = np.meshgrid(x, y)

# Initialize arrays
u = np.ones((Ny, Nx))
v = np.ones((Ny, Nx))

# Set initial conditions
u[(Y >= 0.5) & (Y <= 1.0) & (X >= 0.5) & (X <= 1.0)] = 2.0
v[(Y >= 0.5) & (Y <= 1.0) & (X >= 0.5) & (X <= 1.0)] = 2.0

# Arrays for next time step
un = u.copy()
vn = v.copy()

# Time stepping
for n in range(Nt):
    # Store previous values
    un = u.copy()
    vn = v.copy()
    
    # Update interior points
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):
            # Central difference for spatial derivatives
            u[i,j] = un[i,j] - dt * (
                un[i,j] * (un[i,j+1] - un[i,j-1])/(2*dx) +
                vn[i,j] * (un[i+1,j] - un[i-1,j])/(2*dy)
            )
            
            v[i,j] = vn[i,j] - dt * (
                un[i,j] * (vn[i,j+1] - vn[i,j-1])/(2*dx) +
                vn[i,j] * (vn[i+1,j] - vn[i-1,j])/(2*dy)
            )
    
    # Apply boundary conditions
    u[0,:] = 1.0  # Bottom
    u[-1,:] = 1.0 # Top
    u[:,0] = 1.0  # Left
    u[:,-1] = 1.0 # Right
    
    v[0,:] = 1.0  # Bottom
    v[-1,:] = 1.0 # Top
    v[:,0] = 1.0  # Left
    v[:,-1] = 1.0 # Right

# Plot final results
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.contourf(X, Y, u, levels=np.linspace(1, 2, 20))
plt.colorbar(label='u')
plt.title('u-velocity at t = 2')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(122)
plt.contourf(X, Y, v, levels=np.linspace(1, 2, 20))
plt.colorbar(label='v')
plt.title('v-velocity at t = 2')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()
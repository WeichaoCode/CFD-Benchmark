import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
Nx = 101
Ny = 101
Lx = 2.0
Ly = 2.0
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Time parameters
T = 2.0
nt = 500
dt = T/nt

# Initialize u and v
u = np.ones((Ny, Nx))
v = np.ones((Ny, Nx))

# Set initial conditions
mask = (X >= 0.5) & (X <= 1.0) & (Y >= 0.5) & (Y <= 1.0)
u[mask] = 2.0
v[mask] = 2.0

# Function to apply boundary conditions
def apply_bc(u, v):
    u[0,:] = 1.0  # Bottom
    u[-1,:] = 1.0  # Top
    u[:,0] = 1.0   # Left
    u[:,-1] = 1.0  # Right
    
    v[0,:] = 1.0   # Bottom
    v[-1,:] = 1.0  # Top
    v[:,0] = 1.0   # Left
    v[:,-1] = 1.0  # Right
    return u, v

# Time stepping
for n in range(nt):
    # Store previous values
    u_prev = u.copy()
    v_prev = v.copy()
    
    # Calculate spatial derivatives using central differences
    dudx = np.zeros_like(u)
    dudy = np.zeros_like(u)
    dvdx = np.zeros_like(v)
    dvdy = np.zeros_like(v)
    
    dudx[:,1:-1] = (u_prev[:,2:] - u_prev[:,:-2])/(2*dx)
    dudy[1:-1,:] = (u_prev[2:,:] - u_prev[:-2,:])/(2*dy)
    dvdx[:,1:-1] = (v_prev[:,2:] - v_prev[:,:-2])/(2*dx)
    dvdy[1:-1,:] = (v_prev[2:,:] - v_prev[:-2,:])/(2*dy)
    
    # Update u and v
    u[1:-1,1:-1] = u_prev[1:-1,1:-1] - dt*(u_prev[1:-1,1:-1]*dudx[1:-1,1:-1] + 
                                          v_prev[1:-1,1:-1]*dudy[1:-1,1:-1])
    v[1:-1,1:-1] = v_prev[1:-1,1:-1] - dt*(u_prev[1:-1,1:-1]*dvdx[1:-1,1:-1] + 
                                          v_prev[1:-1,1:-1]*dvdy[1:-1,1:-1])
    
    # Apply boundary conditions
    u, v = apply_bc(u, v)

# Plot final results
plt.figure(figsize=(12,5))

plt.subplot(121)
plt.contourf(X, Y, u, levels=50)
plt.colorbar(label='u')
plt.title('u-velocity at t = {}'.format(T))
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(122)
plt.contourf(X, Y, v, levels=50)
plt.colorbar(label='v')
plt.title('v-velocity at t = {}'.format(T))
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 2.0, 2.0
Nx, Ny = 101, 101
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
nu = 0.05
T = 2.0
nt = 500
dt = T/nt

# Grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initialize solution array
u = np.ones((Ny, Nx))

# Set initial condition
u[(y>=0.5) & (y<=1.0)][:, (x>=0.5) & (x<=1.0)] = 2.0

# Time stepping coefficients
rx = nu*dt/dx**2
ry = nu*dt/dy**2

# Time stepping
for n in range(nt):
    un = u.copy()
    
    # Interior points
    u[1:-1,1:-1] = un[1:-1,1:-1] + \
                    rx*(un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2]) + \
                    ry*(un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1])
    
    # Boundary conditions
    u[0,:] = 1  # Bottom
    u[-1,:] = 1 # Top
    u[:,0] = 1  # Left
    u[:,-1] = 1 # Right

# Plot final solution
plt.figure(figsize=(8,6))
plt.contourf(X, Y, u, levels=50)
plt.colorbar(label='u')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'2D Diffusion at t = {T}')
plt.show()
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

# Initialize solution
u = np.ones((Ny, Nx))
u[(y>=0.5) & (y<=1.0)][:, (x>=0.5) & (x<=1.0)] = 2.0

# Time stepping
for n in range(nt):
    un = u.copy()
    
    # Interior points
    u[1:-1, 1:-1] = un[1:-1, 1:-1] + nu*dt*(
        (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2])/dx**2 +
        (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1])/dy**2
    )
    
    # Boundary conditions
    u[0, :] = 1.0  # Bottom
    u[-1, :] = 1.0  # Top
    u[:, 0] = 1.0  # Left
    u[:, -1] = 1.0  # Right

# Plot final solution
plt.figure(figsize=(8, 8))
plt.contourf(X, Y, u, levels=50, cmap='hot')
plt.colorbar(label='u')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'2D Diffusion at t = {T}')
plt.show()

# Plot centerline
plt.figure(figsize=(8, 6))
plt.plot(x, u[Ny//2, :], 'b-', label='y = 1')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Centerline Profile')
plt.grid(True)
plt.legend()
plt.show()
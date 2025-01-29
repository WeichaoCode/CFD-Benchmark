import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
nx = 101
ny = 101
nt = 500
c = 1.0

# Domain
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

# Time parameters
T = 2.0
dt = T/nt
CFL = c*dt/dx

# Initialize solution array
u = np.ones((ny, nx))
X, Y = np.meshgrid(x, y)

# Set initial condition
u[(Y >= 0.5) & (Y <= 1.0) & (X >= 0.5) & (X <= 1.0)] = 2.0

# Time stepping
un = np.zeros((ny, nx))

for n in range(nt):
    un = u.copy()
    
    # Interior points
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] - 
                     c * dt/dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     c * dt/dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]))
    
    # Boundary conditions
    u[0, :] = 1.0  # Bottom
    u[-1, :] = 1.0  # Top
    u[:, 0] = 1.0  # Left
    u[:, -1] = 1.0  # Right

# Plot final solution
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, u, levels=np.linspace(1, 2, 21))
plt.colorbar(label='u')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'2D Linear Convection at t = {T}')
plt.show()
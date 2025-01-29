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

# Initialize solution array
p = np.zeros((Ny, Nx))

# Set initial conditions
p[:,:] = 0

# Time stepping
for n in range(nt):
    p_old = p.copy()
    
    # Interior points
    for i in range(1, Ny-1):
        for j in range(1, Nx-1):
            p[i,j] = ((p_old[i+1,j] + p_old[i-1,j])/dy**2 + 
                      (p_old[i,j+1] + p_old[i,j-1])/dx**2) / (2/dx**2 + 2/dy**2)
    
    # Boundary conditions
    p[:,0] = 0  # x = 0
    p[:,-1] = y  # x = 2
    p[0,1:-1] = p[1,1:-1]  # y = 0, Neumann
    p[-1,1:-1] = p[-2,1:-1]  # y = 2, Neumann

# Plot results
plt.figure(figsize=(10,8))
plt.contourf(X, Y, p, levels=20, cmap='viridis')
plt.colorbar(label='p')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Laplace Equation Solution')
plt.show()

# Plot surface
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, p, cmap='viridis')
plt.colorbar(surf)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('p')
plt.title('2D Laplace Equation Solution (Surface)')
plt.show()
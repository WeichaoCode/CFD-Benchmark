import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
nx = 101
ny = 101
nt = 500

# Domain
Lx = 2.0
Ly = 2.0
dx = Lx/(nx-1)
dy = Ly/(ny-1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Time parameters
T = 2.0
dt = T/nt

# Initialize arrays
p = np.zeros((ny, nx))
pn = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Source terms
b[ny//4, nx//4] = 100
b[3*ny//4, 3*nx//4] = -100

# Coefficients for the numerical scheme
dx2 = dx*dx
dy2 = dy*dy
dt2 = dt*dt

# Time stepping
for n in range(nt):
    pn = p.copy()
    
    # Interior points
    p[1:-1,1:-1] = ((dx2*dy2)/(2*(dx2+dy2))) * \
                    ((pn[1:-1,2:] + pn[1:-1,:-2])/dx2 + \
                     (pn[2:,1:-1] + pn[:-2,1:-1])/dy2 - \
                      b[1:-1,1:-1])
    
    # Boundary conditions
    p[0,:] = 0  # Bottom
    p[-1,:] = 0 # Top
    p[:,0] = 0  # Left
    p[:,-1] = 0 # Right

# Plot results
plt.figure(figsize=(10,8))
plt.contourf(X, Y, p, levels=50)
plt.colorbar(label='p')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Poisson Equation Solution')
plt.show()

# Plot surface
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, p, cmap='viridis')
plt.colorbar(surf)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('p')
plt.title('2D Poisson Equation Solution (Surface)')
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Problem parameters
Lx = 2
Ly = 1
nx = 100
ny = 50

# Grid spacing
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Initialize solution
p = np.zeros((ny,nx))

# Boundary conditions
p[:,0] = 0              # p = 0 at x = 0 
p[:,-1] = np.linspace(0, Ly, ny) # p = y at x = Lx 
p[0,:] = p[1,:]                 # dp/dy = 0 at y = 0
p[-1,:] = p[-2,:]               # dp/dy = 0 at y = Ly

# Iterative solver
while True:
    pn = p.copy()
    # Five-point difference operator
    p[1:-1,1:-1] = ((dy**2 * (pn[1:-1,2:] + pn[1:-1,:-2]) +
                     dx**2 * (pn[2:,1:-1] + pn[:-2,1:-1])) /
                    (2 * (dx**2 + dy**2)))

    # Enforce boundary conditions
    p[:,0] = 0
    p[:,-1] = np.linspace(0, Ly, ny)
    p[0,:] = p[1,:]
    p[-1,:] = p[-2,:]

    # Check for convergence
    if np.sqrt(np.sum((p - pn)**2)/np.sum(pn**2)) < 1e-6:
        break

# Plot the solution
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
plt.contourf(X, Y, p, cmap='inferno')
plt.title('Distribution of p(x,y)')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='p')
plt.show()
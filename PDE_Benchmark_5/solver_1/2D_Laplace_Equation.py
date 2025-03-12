import numpy as np
import matplotlib.pyplot as plt

def laplace_solver(Lx, Ly, nx, ny, tol=1e-5):
    # Initialize solution and setup grid
    p = np.zeros((ny, nx))
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    dx, dy = x[1]-x[0], y[1]-y[0]
    
    # Set boundary conditions
    p[:,0] = 0
    p[:,-1] = y
    err = 1e3  # Error start guess
    
    # Iteratively solve the 2D Laplace equation (here using Gauss-Seidel method)
    while err > tol:
        pn = p.copy()
        # Finite difference scheme - 5 point stencil
        p[1:-1,1:-1] = (dy**2*(pn[1:-1,2:]+pn[1:-1,:-2]) + dx**2*(pn[2:,1:-1]+pn[:-2,1:-1])) / (2*(dx**2+dy**2))
        
        # Boundary conditions - Specific to this problem
        p[0,:] = p[1,:]
        p[-1,:] = p[-2,:]
        
        # Error calculation
        err = (np.sum(np.abs(p[:]) - np.abs(pn[:])) / np.sum(np.abs(pn[:])))
    return p
  
# Generate values for x and y space
Lx, Ly = 5.0, 5.0
nx, ny = 100, 100
p = laplace_solver(Lx, Ly, nx, ny)

# Plotting the solution
plt.figure(figsize=(8,6))
plt.contourf(p, cmap=plt.cm.viridis)
plt.title('Steady state distribution of p(x,y)')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()
#!/usr/bin/env python3
import numpy as np

# Parameters
nx = 101            # number of grid points in x-direction
ny = 51             # number of grid points in y-direction
Lx = 2.0            # domain length in x-direction
Ly = 1.0            # domain length in y-direction

dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Initialize potential field p
p = np.zeros((ny, nx))  # using (ny, nx) ordering: first index y, second index x

# Set boundary conditions
# Left boundary: x=0 -> p = 0
p[:, 0] = 0.0
# Right boundary: x=2 -> p = y (for each y value)
p[:, -1] = y[:]

# Tolerance and maximum iterations for iterative solver
tol = 1e-5
max_iter = 10000

# Iterative solver using Gauss-Seidel method
for it in range(max_iter):
    p_old = p.copy()
    
    # Enforce Neumann boundary conditions for top and bottom:
    # Bottom: y=0 => dp/dy=0, so p[0,:] = p[1,:]
    p[0, :] = p[1, :]
    # Top: y=Ly => dp/dy=0, so p[-1,:] = p[-2,:]
    p[-1, :] = p[-2, :]
    
    # Update interior points
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            p[j, i] = 0.5 * ((p[j, i+1] + p[j, i-1]) * dy**2 + (p[j+1, i] + p[j-1, i]) * dx**2) / (dx**2 + dy**2)
    
    # Reimpose Dirichlet BC on left and right boundaries:
    p[:, 0] = 0.0
    p[:, -1] = y[:]  # right boundary: p = y
    
    # Check convergence
    diff = np.linalg.norm(p - p_old, ord=np.inf)
    if diff < tol:
        break

# Save the final solution as a 2D NumPy array in a file named "p.npy"
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/p_2D_Laplace_Equation.npy', p)
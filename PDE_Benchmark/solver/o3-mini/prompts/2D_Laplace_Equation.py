#!/usr/bin/env python3
import numpy as np

# Domain parameters
Lx = 2.0
Ly = 1.0
Nx = 101  # number of grid points in x
Ny = 51   # number of grid points in y

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# Create grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

# Initialize p field with initial condition p=0 everywhere
p = np.zeros((Nx, Ny))

# Apply Dirichlet boundary conditions
# Left boundary: x=0, p=0
p[0, :] = 0.0
# Right boundary: x = Lx, p = y
p[-1, :] = y[:]  # broadcasting of y coordinate to right boundary

# Define relaxation parameters and convergence criteria
tol = 1e-6
max_iter = 10000

# Precompute coefficients for the iterative update using central differences
dx2 = dx**2
dy2 = dy**2
coef = 1.0 / (2.0*(dx2 + dy2))

for iteration in range(max_iter):
    p_old = p.copy()

    # Update interior points: for i=1 to Nx-2 and j=1 to Ny-2
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            p[i, j] = coef * (dy2*(p[i+1, j] + p[i-1, j]) + dx2*(p[i, j+1] + p[i, j-1]))
    
    # Enforce Neumann boundary conditions on top and bottom (y=0 and y=Ly)
    # Bottom boundary data: j=0, use central difference -> p[i,0] = p[i,1]
    for i in range(1, Nx-1):
        p[i, 0] = p[i, 1]
    # Top boundary data: j=Ny-1, use central difference -> p[i,Ny-1] = p[i,Ny-2]
    for i in range(1, Nx-1):
        p[i, -1] = p[i, -2]

    # Reapply Dirichlet boundary conditions on left and right boundaries
    p[0, :] = 0.0
    p[-1, :] = y[:]

    # Check for convergence
    if np.max(np.abs(p - p_old)) < tol:
        break

# Save final solution as a 2D numpy array in 'p.npy'
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/p_2D_Laplace_Equation.npy', p)
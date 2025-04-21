import numpy as np

# Grid parameters
nx = 101  # Number of points in x
ny = 51   # Number of points in y
Lx = 2.0  # Domain length in x
Ly = 1.0  # Domain length in y
dx = Lx/(nx-1)
dy = Ly/(ny-1)

# Initialize arrays
p = np.zeros((ny,nx))
b = np.zeros((ny,nx))

# Set source term b
source_x1 = int(0.25*Lx/dx)
source_y1 = int(0.25*Ly/dy)
source_x2 = int(0.75*Lx/dx)
source_y2 = int(0.75*Ly/dy)
b[source_y1,source_x1] = 100
b[source_y2,source_x2] = -100

# Iteration parameters
max_iter = 10000
tolerance = 1e-6

# Gauss-Seidel iteration
for it in range(max_iter):
    p_old = p.copy()
    
    # Update interior points
    for i in range(1,ny-1):
        for j in range(1,nx-1):
            p[i,j] = 0.25*(p[i+1,j] + p[i-1,j] + p[i,j+1] + p[i,j-1] - dx*dy*b[i,j])
    
    # Check convergence
    error = np.max(np.abs(p - p_old))
    if error < tolerance:
        break

# Save solution
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/p_2D_Poisson_Equation.npy', p)
import numpy as np

# Grid parameters
nx = 100
ny = 50
dx = 2.0 / (nx-1)
dy = 1.0 / (ny-1)

# Initialize solution array
p = np.zeros((ny, nx))

# Set right boundary condition
x = np.linspace(0, 2, nx)
y = np.linspace(0, 1, ny)
p[:,-1] = y

# Iteration parameters
max_iter = 10000
tolerance = 1e-6

# Gauss-Seidel iteration
for it in range(max_iter):
    p_old = p.copy()
    
    # Update interior points
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            p[j,i] = 0.25*(p[j,i+1] + p[j,i-1] + p[j+1,i] + p[j-1,i])
    
    # Neumann boundary conditions (top/bottom)
    p[0,:] = p[1,:]  # Bottom
    p[-1,:] = p[-2,:] # Top
    
    # Fixed boundary conditions
    p[:,0] = 0  # Left
    p[:,-1] = y # Right
    
    # Check convergence
    error = np.max(np.abs(p - p_old))
    if error < tolerance:
        break

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/p_2D_Laplace_Equation.npy', p)
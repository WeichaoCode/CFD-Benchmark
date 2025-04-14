import numpy as np

# Grid parameters
nx = 51  # Number of points in x direction
ny = 41  # Number of points in y direction
dx = 5.0/(nx-1)  # Grid spacing in x
dy = 4.0/(ny-1)  # Grid spacing in y

# Initialize temperature field
T = np.zeros((ny, nx))

# Set boundary conditions
T[0, :] = 20  # Bottom boundary
T[-1, :] = 0  # Top boundary 
T[:, 0] = 10  # Left boundary
T[:, -1] = 40  # Right boundary

# Iteration parameters
max_iter = 10000
tolerance = 1e-6

# Gauss-Seidel iteration
for it in range(max_iter):
    T_old = T.copy()
    
    # Update interior points
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            T[i,j] = 0.25*(T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1])
    
    # Check convergence
    error = np.max(np.abs(T - T_old))
    if error < tolerance:
        break

# Save temperature field
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/T_2D_Steady_Heat_Equation.npy', T)
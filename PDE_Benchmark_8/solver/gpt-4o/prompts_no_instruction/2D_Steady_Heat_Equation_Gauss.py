import numpy as np

# Domain parameters
width = 5.0
height = 4.0
dx = 0.05
dy = 0.05
nx = int(width / dx) + 1
ny = int(height / dy) + 1

# Initialize temperature field
T = np.zeros((ny, nx))

# Boundary conditions
T[:, 0] = 10.0  # Left boundary (x = 0)
T[:, -1] = 40.0  # Right boundary (x = 5)
T[-1, :] = 0.0  # Top boundary (y = 4)
T[0, :] = 20.0  # Bottom boundary (y = 0)

# Gauss-Seidel iteration parameters
tolerance = 1e-5
max_iterations = 10000

# Gauss-Seidel iteration
for iteration in range(max_iterations):
    T_old = T.copy()
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T[j, i] = 0.25 * (T[j+1, i] + T[j-1, i] + T[j, i+1] + T[j, i-1])
    
    # Check for convergence
    if np.linalg.norm(T - T_old, ord=np.inf) < tolerance:
        break

# Save the final temperature field
save_values = ['T']
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/T_2D_Steady_Heat_Equation_Gauss.npy', T)
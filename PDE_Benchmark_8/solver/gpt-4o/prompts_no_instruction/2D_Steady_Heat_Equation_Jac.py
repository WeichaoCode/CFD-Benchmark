import numpy as np

# Parameters
width = 5.0
height = 4.0
dx = 0.05
dy = 0.05
nx = int(width / dx) + 1
ny = int(height / dy) + 1
tolerance = 1e-6
max_iterations = 10000

# Initialize temperature field
T = np.zeros((ny, nx))

# Apply boundary conditions
T[:, 0] = 10.0  # Left boundary (x = 0)
T[:, -1] = 40.0  # Right boundary (x = 5)
T[-1, :] = 0.0  # Top boundary (y = 4)
T[0, :] = 20.0  # Bottom boundary (y = 0)

# Jacobi iteration
T_new = np.copy(T)
for iteration in range(max_iterations):
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T_new[j, i] = 0.25 * (T[j+1, i] + T[j-1, i] + T[j, i+1] + T[j, i-1])
    
    # Check for convergence
    if np.linalg.norm(T_new - T, ord=np.inf) < tolerance:
        break
    
    T[:, :] = T_new[:, :]

# Save the final solution
save_values = ['T']
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/T_2D_Steady_Heat_Equation_Jac.npy', T)
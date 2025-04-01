import numpy as np

# Grid parameters
nx = 101
ny = 81
dx = 0.05
dy = 0.05

# Initialize temperature field
T = np.zeros((ny, nx))

# Apply boundary conditions
T[:, 0] = 10      # Left boundary (x=0)
T[:, -1] = 40     # Right boundary (x=5)
T[0, :] = 20      # Bottom boundary (y=0)
T[-1, :] = 0      # Top boundary (y=4)

# Initialize variables for Jacobi iteration
T_new = T.copy()
tolerance = 1e-4
max_iterations = 10000

# Jacobi iteration loop
for _ in range(max_iterations):
    T_new[1:-1, 1:-1] = 0.25 * (T[0:-2, 1:-1] + T[2:, 1:-1] + T[1:-1, 0:-2] + T[1:-1, 2:])
    diff = np.max(np.abs(T_new - T))
    if diff < tolerance:
        break
    T[:, :] = T_new

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/T_2D_Steady_Heat_Equation_Jac.npy', T)
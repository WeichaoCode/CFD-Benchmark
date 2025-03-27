import numpy as np

# Define the domain and grid
Lx, Ly = 5.0, 4.0
dx, dy = 0.05, 0.05
nx, ny = int(Lx/dx) + 1, int(Ly/dy) + 1

# Initialize the temperature field
T = np.zeros((ny, nx))

# Apply boundary conditions
T[:, 0] = 10.0  # Left boundary
T[:, -1] = 40.0  # Right boundary
T[0, :] = 0.0  # Top boundary
T[-1, :] = 20.0  # Bottom boundary

# Gauss-Seidel iteration parameters
tolerance = 1e-5
max_iterations = 10000

# Gauss-Seidel iteration
for iteration in range(max_iterations):
    T_old = T.copy()
    
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T[j, i] = 0.25 * (T_old[j+1, i] + T[j-1, i] + T[j, i+1] + T[j, i-1])
    
    # Check for convergence
    diff = np.linalg.norm(T - T_old, ord=np.inf)
    if diff < tolerance:
        print(f"Converged after {iteration+1} iterations.")
        break
else:
    print("Did not converge within the maximum number of iterations.")

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/T_2D_Steady_Heat_Equation_Gauss.npy', T)
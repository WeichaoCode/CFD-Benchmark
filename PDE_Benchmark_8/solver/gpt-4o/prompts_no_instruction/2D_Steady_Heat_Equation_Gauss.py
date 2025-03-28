import numpy as np

# Domain parameters
Lx, Ly = 5.0, 4.0
dx, dy = 0.05, 0.05
nx, ny = 101, 81

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Initialize temperature field
T = np.zeros((ny, nx))

# Boundary conditions
T[:, 0] = 10.0  # Left boundary
T[:, -1] = 40.0  # Right boundary
T[0, :] = 0.0  # Top boundary
T[-1, :] = 20.0  # Bottom boundary

# Gauss-Seidel iteration parameters
tolerance = 1e-6
max_iterations = 10000

# Gauss-Seidel iteration
for iteration in range(max_iterations):
    T_old = T.copy()
    
    # Update the interior points
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T[j, i] = 0.25 * (T[j+1, i] + T[j-1, i] + T[j, i+1] + T[j, i-1])
    
    # Check for convergence
    diff = np.linalg.norm(T - T_old, ord=np.inf)
    if diff < tolerance:
        print(f"Converged after {iteration+1} iterations.")
        break
else:
    print("Did not converge within the maximum number of iterations.")

# Save the final temperature field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/T_2D_Steady_Heat_Equation_Gauss.npy', T)
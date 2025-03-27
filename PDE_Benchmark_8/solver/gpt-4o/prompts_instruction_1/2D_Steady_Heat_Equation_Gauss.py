import numpy as np

# Define the domain and grid parameters
Lx, Ly = 5.0, 4.0  # Length of the domain in x and y directions
dx, dy = 0.05, 0.05  # Grid spacing
nx, ny = int(Lx/dx) + 1, int(Ly/dy) + 1  # Number of grid points

# Initialize the temperature field
T = np.zeros((ny, nx))

# Set the boundary conditions
T[:, 0] = 10.0  # Left boundary
T[:, -1] = 40.0  # Right boundary
T[0, :] = 0.0  # Top boundary
T[-1, :] = 20.0  # Bottom boundary

# Gauss-Seidel iteration parameters
beta = dx / dy
tolerance = 1e-4
max_iterations = 10000

# Gauss-Seidel iteration
for iteration in range(max_iterations):
    T_old = T.copy()
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            T[j, i] = (T_old[j, i+1] + T[j, i-1] + beta**2 * (T[j+1, i] + T[j-1, i])) / (2 * (1 + beta**2))
    
    # Enforce boundary conditions
    T[:, 0] = 10.0
    T[:, -1] = 40.0
    T[0, :] = 0.0
    T[-1, :] = 20.0
    
    # Check for convergence
    if np.max(np.abs(T - T_old)) < tolerance:
        print(f"Converged after {iteration+1} iterations.")
        break
else:
    print("Did not converge within the maximum number of iterations.")

# Save the final temperature distribution to a .npy file
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_1/T_2D_Steady_Heat_Equation_Gauss.npy', T)
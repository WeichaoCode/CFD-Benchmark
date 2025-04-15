import numpy as np

# Domain parameters
Lx, Ly = 5, 4  # Domain dimensions
nx, ny = 100, 80  # Grid resolution

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize temperature field
T = np.zeros((ny, nx))

# Apply boundary conditions
T[0, :] = 20  # Bottom boundary
T[-1, :] = 0  # Top boundary
T[:, 0] = 10  # Left boundary
T[:, -1] = 40  # Right boundary

# Solve using finite difference method (Laplace equation)
def solve_laplace(T, max_iter=10000, tolerance=1e-4):
    T_new = T.copy()
    for _ in range(max_iter):
        T_prev = T_new.copy()
        
        # Gauss-Seidel iteration
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                T_new[i, j] = 0.25 * (T_new[i+1, j] + T_new[i-1, j] + 
                                       T_new[i, j+1] + T_new[i, j-1])
        
        # Check convergence
        if np.max(np.abs(T_new - T_prev)) < tolerance:
            break
    
    return T_new

# Solve the Laplace equation
T_final = solve_laplace(T)

# Save the final temperature field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts/T_final_2D_Steady_Heat_Equation.npy', T_final)
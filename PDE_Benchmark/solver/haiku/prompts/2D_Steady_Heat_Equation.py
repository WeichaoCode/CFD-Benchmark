import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
Lx, Ly = 5, 4
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

# Solve using finite difference method (Poisson equation)
def solve_laplace(T):
    # Copy boundary conditions
    T_new = T.copy()
    
    # Iterative solution using Jacobi method
    max_iter = 10000
    tolerance = 1e-4
    
    for _ in range(max_iter):
        T_old = T_new.copy()
        
        # Update interior points
        T_new[1:-1, 1:-1] = 0.25 * (
            T_old[1:-1, 2:] +   # Right neighbor
            T_old[1:-1, :-2] +  # Left neighbor
            T_old[2:, 1:-1] +   # Bottom neighbor
            T_old[:-2, 1:-1]    # Top neighbor
        )
        
        # Check convergence
        if np.max(np.abs(T_new - T_old)) < tolerance:
            break
    
    return T_new

# Solve the Laplace equation
T_solution = solve_laplace(T)

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/T_solution_2D_Steady_Heat_Equation.npy', T_solution)
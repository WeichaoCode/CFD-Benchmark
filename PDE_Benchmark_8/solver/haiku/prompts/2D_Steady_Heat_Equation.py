import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
Lx, Ly = 5.0, 4.0
nx, ny = 100, 80  # Grid resolution

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize temperature field
T = np.zeros((ny, nx))

# Apply boundary conditions
T[0, :] = 20.0  # Bottom boundary
T[-1, :] = 0.0  # Top boundary
T[:, 0] = 10.0  # Left boundary 
T[:, -1] = 40.0  # Right boundary

# Solve using finite difference method (Poisson equation)
def poisson_solve(T):
    # Create copy of T for iteration
    T_new = T.copy()
    
    # Solve using Gauss-Seidel iteration
    max_iter = 10000
    tolerance = 1e-6
    
    for _ in range(max_iter):
        # Store previous iteration
        T_prev = T_new.copy()
        
        # Update interior points using finite difference
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                T_new[i,j] = 0.25 * (T_new[i+1,j] + T_new[i-1,j] + 
                                      T_new[i,j+1] + T_new[i,j-1])
        
        # Check convergence
        if np.max(np.abs(T_new - T_prev)) < tolerance:
            break
    
    return T_new

# Solve the Poisson equation
T_solution = poisson_solve(T)

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/haiku/prompts/T_solution_2D_Steady_Heat_Equation.npy', T_solution)
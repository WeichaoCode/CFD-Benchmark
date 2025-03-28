import numpy as np

# Domain parameters
nx, ny = 31, 31
dx = 2 / (nx - 1)
dy = 1 / (ny - 1)

# Initialize the potential field
p = np.zeros((ny, nx))

# Boundary conditions
# Left boundary (x = 0): p = 0
p[:, 0] = 0

# Right boundary (x = 2): p = y
p[:, -1] = np.linspace(0, 1, ny)

# Neumann boundary conditions for top and bottom (y = 0, y = 1)
# These will be handled in the iteration loop

# Iteration parameters
tolerance = 1e-5
max_iterations = 10000

# Iterative solver using the finite difference method
for iteration in range(max_iterations):
    p_old = p.copy()
    
    # Update the interior points
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            p[j, i] = 0.25 * (p_old[j, i+1] + p_old[j, i-1] + p_old[j+1, i] + p_old[j-1, i])
    
    # Apply Neumann boundary conditions (zero gradient) for top and bottom
    p[0, :] = p[1, :]  # Bottom boundary (y = 0)
    p[-1, :] = p[-2, :]  # Top boundary (y = 1)
    
    # Check for convergence
    if np.linalg.norm(p - p_old, ord=np.inf) < tolerance:
        print(f"Converged after {iteration} iterations.")
        break
else:
    print("Did not converge within the maximum number of iterations.")

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/p_2D_Laplace_Equation.npy', p)
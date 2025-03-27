import numpy as np

# Define the grid
nx, ny = 31, 31
dx = 2 / (nx - 1)
dy = 1 / (ny - 1)

# Initialize the potential field
p = np.zeros((ny, nx))

# Boundary conditions
p[:, 0] = 0  # Left boundary
p[:, -1] = np.linspace(0, 1, ny)  # Right boundary

# Iterative solver parameters
tolerance = 1e-5
max_iterations = 10000

# Iterative solver using the finite difference method
for iteration in range(max_iterations):
    p_old = p.copy()
    
    # Update the interior points
    p[1:-1, 1:-1] = ((p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dy**2 +
                     (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dx**2) / (2 * (dx**2 + dy**2))
    
    # Neumann boundary conditions (top and bottom)
    p[0, :] = p[1, :]  # Top boundary
    p[-1, :] = p[-2, :]  # Bottom boundary
    
    # Check for convergence
    if np.linalg.norm(p - p_old, ord=np.inf) < tolerance:
        print(f"Converged after {iteration} iterations.")
        break
else:
    print("Did not converge within the maximum number of iterations.")

# Save the final solution
np.save('solution.npy', p)
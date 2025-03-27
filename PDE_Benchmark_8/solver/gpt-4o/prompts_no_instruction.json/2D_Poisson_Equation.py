import numpy as np

# Define the domain and grid
Lx, Ly = 2.0, 1.0
nx, ny = 50, 50
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Initialize the potential field p and the source term b
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Set the source term b according to the problem statement
b[int(ny / 4), int(nx / 4)] = 100
b[int(3 * ny / 4), int(3 * nx / 4)] = -100

# Define the number of iterations for the iterative solver
num_iterations = 10000
tolerance = 1e-5

# Iterative solver using the Jacobi method
for it in range(num_iterations):
    p_old = p.copy()
    
    # Update the potential field p using the finite difference method
    p[1:-1, 1:-1] = ((p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dy**2 +
                     (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dx**2 -
                     b[1:-1, 1:-1] * dx**2 * dy**2) / (2 * (dx**2 + dy**2))
    
    # Apply Dirichlet boundary conditions
    p[:, 0] = 0
    p[:, -1] = 0
    p[0, :] = 0
    p[-1, :] = 0
    
    # Check for convergence
    if np.linalg.norm(p - p_old, ord=np.inf) < tolerance:
        print(f'Converged after {it} iterations')
        break

# Save the final solution to a .npy file
np.save('solution.npy', p)
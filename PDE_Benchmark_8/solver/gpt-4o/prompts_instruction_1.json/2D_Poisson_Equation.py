import numpy as np

# Domain parameters
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

# Iterative solver parameters
tolerance = 1e-5
max_iterations = 10000

# Jacobi iterative solver
for iteration in range(max_iterations):
    p_old = p.copy()
    
    # Update p using the Jacobi method
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            p[j, i] = ((p_old[j, i + 1] + p_old[j, i - 1]) * dy**2 +
                       (p_old[j + 1, i] + p_old[j - 1, i]) * dx**2 -
                       b[j, i] * dx**2 * dy**2) / (2 * (dx**2 + dy**2))
    
    # Apply Dirichlet boundary conditions
    p[0, :] = 0
    p[-1, :] = 0
    p[:, 0] = 0
    p[:, -1] = 0
    
    # Check for convergence
    if np.linalg.norm(p - p_old, ord=np.inf) < tolerance:
        print(f"Converged after {iteration} iterations.")
        break
else:
    print("Did not converge within the maximum number of iterations.")

# Save the final solution to a .npy file
np.save('solution.npy', p)
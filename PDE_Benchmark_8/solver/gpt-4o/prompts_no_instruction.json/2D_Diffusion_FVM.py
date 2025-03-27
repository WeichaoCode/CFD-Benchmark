import numpy as np

# Parameters
mu = 1e-3  # dynamic viscosity (Pa·s)
dP_dz = -3.2  # pressure gradient (Pa/m)
h = 0.1  # domain height (m)
n_x = n_y = 80  # number of grid points
dx = dy = h / (n_x - 1)  # grid spacing

# Initialize the velocity field
w = np.zeros((n_x, n_y))

# Coefficients for the finite volume method
A = mu / dx**2
B = mu / dy**2
C = -dP_dz

# Iterative solver parameters
tolerance = 1e-6
max_iterations = 10000

# Iterative solver (Gauss-Seidel method)
for iteration in range(max_iterations):
    w_old = w.copy()
    
    for i in range(1, n_x - 1):
        for j in range(1, n_y - 1):
            w[i, j] = (A * (w_old[i+1, j] + w_old[i-1, j]) +
                       B * (w_old[i, j+1] + w_old[i, j-1]) +
                       C * dx * dy) / (2 * (A + B))
    
    # Check for convergence
    if np.linalg.norm(w - w_old, ord=np.inf) < tolerance:
        print(f"Converged after {iteration} iterations.")
        break
else:
    print("Did not converge within the maximum number of iterations.")

# Save the final solution
np.save('w_solution.npy', w)
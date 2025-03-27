import numpy as np

# Parameters
H = 2.0  # Height of the domain
n = 100  # Number of grid points
mu = 1.0  # Molecular viscosity (assumed constant for simplicity)

# Create a non-uniform grid clustered near the walls
y = np.linspace(0, H, n)
dy = np.gradient(y)

# Initialize velocity and effective viscosity arrays
ubar = np.zeros(n)
mu_t = np.zeros(n)  # Turbulent eddy viscosity (initially zero)

# Function to compute turbulent eddy viscosity using Spalart-Allmaras model
def compute_mu_t(ubar, y, mu, mu_t):
    # Placeholder for the Spalart-Allmaras model computation
    # For simplicity, assume a constant eddy viscosity
    return 0.1 * mu * np.ones_like(ubar)

# Iterative solver for the steady-state problem
tolerance = 1e-6
max_iterations = 1000
for iteration in range(max_iterations):
    mu_t = compute_mu_t(ubar, y, mu, mu_t)
    mu_eff = mu + mu_t

    # Construct the finite difference matrix and right-hand side
    A = np.zeros((n, n))
    b = np.zeros(n)

    for i in range(1, n-1):
        A[i, i-1] = mu_eff[i-1] / dy[i-1]**2
        A[i, i] = -(mu_eff[i-1] / dy[i-1]**2 + mu_eff[i] / dy[i]**2)
        A[i, i+1] = mu_eff[i] / dy[i]**2
        b[i] = -1

    # Apply Dirichlet boundary conditions
    A[0, 0] = 1
    A[-1, -1] = 1
    b[0] = 0
    b[-1] = 0

    # Solve the linear system
    ubar_new = np.linalg.solve(A, b)

    # Check for convergence
    if np.linalg.norm(ubar_new - ubar, ord=np.inf) < tolerance:
        print(f"Converged in {iteration} iterations.")
        break

    ubar = ubar_new

# Save the final solution
np.save('final_solution.npy', ubar)
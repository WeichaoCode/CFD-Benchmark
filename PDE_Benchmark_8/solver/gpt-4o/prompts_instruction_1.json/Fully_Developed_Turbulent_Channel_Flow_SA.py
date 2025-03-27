import numpy as np

# Parameters
H = 2.0  # Height of the domain
n = 100  # Number of grid points
rho = 1.0  # Density (assumed constant for simplicity)
mu = 1.0e-3  # Molecular viscosity (assumed constant for simplicity)

# Non-uniform grid clustering near the walls
y = np.linspace(0, H, n)
dy = np.gradient(y)

# Initialize velocity and eddy viscosity
u_initial = np.zeros(n)
nu_t = np.zeros(n)  # Turbulent eddy viscosity

# Spalart-Allmaras model parameters (simplified for demonstration)
def compute_nu_t(y, u):
    # Placeholder for a more complex Spalart-Allmaras model
    # Here we use a simple model for demonstration
    return 0.01 * (1 - y / H) * (y / H) * rho

# Compute effective viscosity
def compute_mu_eff(mu, nu_t):
    return mu + nu_t

# Finite Difference Method to solve the PDE
def solve_pde(y, dy, mu, rho, u_initial):
    # Initialize the linear system
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Fill the matrix A and vector b
    for i in range(1, n-1):
        nu_t[i] = compute_nu_t(y[i], u_initial[i])
        mu_eff = compute_mu_eff(mu, nu_t[i])

        A[i, i-1] = mu_eff / dy[i-1]**2
        A[i, i] = -2 * mu_eff / dy[i]**2
        A[i, i+1] = mu_eff / dy[i+1]**2
        b[i] = -1

    # Apply boundary conditions
    A[0, 0] = 1
    A[-1, -1] = 1
    b[0] = 0
    b[-1] = 0

    # Solve the linear system
    u = np.linalg.solve(A, b)
    return u

# Solve the PDE
u_final = solve_pde(y, dy, mu, rho, u_initial)

# Save the final solution as a .npy file
np.save('velocity_profile.npy', u_final)
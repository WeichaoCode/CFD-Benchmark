import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
H = 2.0
n = 100
mu = 1.0  # Molecular viscosity

# Generate non-uniform grid clustered near the walls
beta = 1.5
xi = np.linspace(0, 1, n)
y = H * (xi - 0.5) + (H / 2) * np.sinh(beta * (xi - 0.5)) / np.sinh(beta / 2)
dy = np.diff(y)  # size n-1

# Spalart-Allmaras model constants
C_b1 = 0.1355
C_w1 = 5.0
C_w2 = 0.3
sigma = 2.0
C_b2 = 0.622
kappa = 0.41

# Initialize variables
u = np.zeros(n)
mu_t = np.ones(n) * 0.1
mu_eff = mu + mu_t

# Iteration parameters
max_iter = 1000
tol = 1e-6
for it in range(max_iter):
    mu_eff = mu + mu_t

    # Compute mu_eff at half nodes
    mu_eff_half = 0.5 * (mu_eff[:-1] + mu_eff[1:])  # size n-1

    # Coefficients for the internal nodes
    a = mu_eff_half[:-1] / dy[:-1]  # size n-2
    c = mu_eff_half[1:] / dy[1:]    # size n-2
    main_diag = a + c                # size n-2
    lower_diag = -a[:-1]             # size n-3
    upper_diag = -c[:-1]             # size n-3

    # Construct sparse matrix
    diagonals = [lower_diag, main_diag, upper_diag]
    A = diags(diagonals, offsets=[-1, 0, 1], shape=(n-2, n-2), format='csr')

    # Right-hand side
    RHS = -np.ones(n-2)

    # Solve linear system
    u_inner = spsolve(A, RHS)

    # Update the solution including boundary conditions
    u_new = np.zeros(n)
    u_new[1:-1] = u_inner
    u_new[0] = 0.0
    u_new[-1] = 0.0

    # Compute du/dy using central differences
    du_dy = np.zeros(n)
    du_dy[1:-1] = (u_new[2:] - u_new[:-2]) / (y[2:] - y[:-2])
    du_dy[0] = (u_new[1] - u_new[0]) / dy[0]
    du_dy[-1] = (u_new[-1] - u_new[-2]) / dy[-1]

    # Update mu_t using the Spalart-Allmaras model
    S = np.abs(du_dy)
    mu_t_new = C_b1 * mu_eff * S / (1 + C_b2 * S)

    # Check convergence
    if np.linalg.norm(u_new - u, np.inf) < tol and np.linalg.norm(mu_t_new - mu_t, np.inf) < tol:
        u = u_new
        mu_t = mu_t_new
        break

    u = u_new
    mu_t = mu_t_new

# Save the final solutions
save_values = {
    'u_bar': u,
    'mu_t': mu_t
}

for var_name, data in save_values.items():
    np.save(f'{var_name}.npy', data)
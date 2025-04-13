import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
Re_tau = 395
mu = 1 / Re_tau
rho = 1.0
L = 2.0
N = 100  # Number of grid points
dy = L / (N - 1)

# Grid
y = np.linspace(0, L, N)

# Initial guess for velocity
u = np.zeros(N)

# Turbulent eddy viscosity model (example: linear profile)
mu_t = 0.01 * (1 - (y / L)**2)

# Effective viscosity
mu_eff = mu + mu_t

# Construct the coefficient matrix A and right-hand side vector b
lower_diag = -mu_eff[1:-1] / dy**2
main_diag = np.zeros(N)
upper_diag = -mu_eff[1:-1] / dy**2

# Fill the main diagonal
main_diag[1:-1] = (mu_eff[1:-1] + mu_eff[2:]) / dy**2

# Apply boundary conditions
main_diag[0] = 1.0
main_diag[-1] = 1.0

# Create the sparse matrix
A = diags(
    [np.append(lower_diag, 0), main_diag, np.append(0, upper_diag)],
    offsets=[-1, 0, 1],
    shape=(N, N)
).tocsc()

# Right-hand side vector
b = np.full(N, -1.0)
b[0] = 0.0
b[-1] = 0.0

# Solve the linear system
u = spsolve(A, b)

# Save the solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/u_Fully_Developed_Turbulent_Channel_Flow.npy', u)
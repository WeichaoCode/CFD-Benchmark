import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
H = 2.0  # Height of the domain
N = 100  # Number of grid points
dy = H / (N - 1)  # Grid spacing
mu = 1.0  # Molecular viscosity

# Cess model parameters
A = 26.0
B = 1.0
kappa = 0.41
Re_tau = 5200.0

# Grid
y = np.linspace(0, H, N)

# Initial condition
ubar = np.zeros(N)

# Cess model for turbulent viscosity
def mu_t(y):
    eta = y / H
    return 0.5 * (1 + np.tanh(A * (eta - 0.5))) * (1 - np.exp(-B * eta)) * (kappa * Re_tau * mu) ** 2

# Effective viscosity
def mu_eff(y):
    return mu + mu_t(y)

# Construct the linear system
main_diag = np.zeros(N)
off_diag = np.zeros(N-1)

for i in range(1, N-1):
    mu_eff_ip = mu_eff(y[i] + 0.5 * dy)
    mu_eff_im = mu_eff(y[i] - 0.5 * dy)
    main_diag[i] = (mu_eff_ip + mu_eff_im) / dy**2
    off_diag[i-1] = -mu_eff_ip / dy**2

# Boundary conditions
main_diag[0] = 1.0
main_diag[-1] = 1.0

# Right-hand side
rhs = -np.ones(N)
rhs[0] = 0.0
rhs[-1] = 0.0

# Sparse matrix
diagonals = [main_diag, off_diag, off_diag]
A_matrix = diags(diagonals, [0, -1, 1], format='csc')

# Solve the linear system
ubar = spsolve(A_matrix, rhs)

# Save the solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts/ubar_Fully_Developed_Turbulent_Channel_Flow.npy', ubar)
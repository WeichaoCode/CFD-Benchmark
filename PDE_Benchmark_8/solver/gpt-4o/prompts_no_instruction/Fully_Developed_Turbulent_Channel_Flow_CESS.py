import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

# Parameters
H = 2.0
n = 100
mu = 1.0  # Molecular viscosity

# Non-uniform grid clustering near the walls
y = np.linspace(0, 1, n)
y = H * (np.sinh(3 * (y - 0.5)) / np.sinh(1.5) + 0.5)

# Compute dy and dy_mid
dy = np.diff(y)
dy_mid = (dy[:-1] + dy[1:]) / 2

# Cess turbulence model parameters
Re_tau = 550.0
A = 26.0
B = 0.5
kappa = 0.41

# Compute turbulent viscosity using Cess model
y_plus = Re_tau * y / H
mu_t = 0.5 * mu * (1 + (A * y_plus * (1 - y_plus / Re_tau) ** 2) ** 2) ** 0.5 - 0.5 * mu

# Effective viscosity
mu_eff = mu + mu_t

# Setup the linear system
A_matrix = np.zeros((n, n))
b_vector = np.zeros(n)

# Fill the matrix and RHS vector
for i in range(1, n-1):
    A_matrix[i, i-1] = (mu_eff[i-1] + mu_eff[i]) / (2 * dy[i-1])
    A_matrix[i, i] = -(mu_eff[i-1] + 2 * mu_eff[i] + mu_eff[i+1]) / (2 * dy_mid[i-1])
    A_matrix[i, i+1] = (mu_eff[i] + mu_eff[i+1]) / (2 * dy[i])
    b_vector[i] = -1

# Apply Dirichlet boundary conditions
A_matrix[0, 0] = 1
A_matrix[-1, -1] = 1
b_vector[0] = 0
b_vector[-1] = 0

# Convert A_matrix to CSR format
A_sparse = csr_matrix(A_matrix)

# Solve the linear system
u = spsolve(A_sparse, b_vector)

# Save the solution
save_values = ['u']
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/u_Fully_Developed_Turbulent_Channel_Flow_CESS.npy', u)
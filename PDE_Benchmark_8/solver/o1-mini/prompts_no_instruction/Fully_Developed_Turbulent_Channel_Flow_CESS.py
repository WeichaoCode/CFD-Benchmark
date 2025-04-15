import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
H = 2.0
n = 100
mu = 1.0
C_mu = 0.09

# Create non-uniform grid clustered near the walls
i = np.arange(n)
y = H * (1 - np.cos(np.pi * i / (n - 1))) / 2

# Compute effective viscosity
mu_t = C_mu * (1 - (y / H)**2)
mu_eff = mu + mu_t

# Compute a at interfaces
a_interface = (mu_eff[:-1] + mu_eff[1:]) / 2

# Compute grid spacing
dy = np.diff(y)

# Coefficients for the tridiagonal matrix
W = a_interface[:-1] / dy[:-1]
E = a_interface[:-1] / dy[:-1]
A = W + E

# Construct the diagonals
main_diag = A
lower_diag = -W
upper_diag = -E

# Assemble the sparse matrix
diagonals = [lower_diag, main_diag, upper_diag]
offsets = [-1, 0, 1]
A_matrix = diags(diagonals, offsets, shape=(n-2, n-2), format='csr')

# Right-hand side
b = -np.ones(n-2)

# Solve the linear system
u_internal = spsolve(A_matrix, b)

# Complete the solution including boundary conditions
u = np.zeros(n)
u[1:-1] = u_internal

# Save the solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts_no_instruction/u_Fully_Developed_Turbulent_Channel_Flow_CESS.npy', u)
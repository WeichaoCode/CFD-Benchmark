import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
H = 2.0  # Height of the domain
n = 100  # Number of grid points
mu = 1.0  # Molecular viscosity

# Cess model parameters
A_plus = 26.0
B = 1.0 / 0.41
Re_tau = 5200.0

# Non-uniform grid clustering near the walls
y = np.linspace(0, 1, n)
y = H * (1 - np.cos(np.pi * y)) / 2

# Compute dy
dy = np.diff(y)

# Initialize velocity field
ubar = np.zeros(n)

# Compute turbulent viscosity using Cess model
def compute_mu_t(y, H, Re_tau, A_plus, B):
    eta = y / H
    return 0.5 * (1 + np.tanh(A_plus * (eta - 0.5))) * (1 - np.exp(-B * eta * Re_tau))

mu_t = compute_mu_t(y, H, Re_tau, A_plus, B)
mu_eff = mu + mu_t

# Construct the finite difference matrix
lower_diag = mu_eff[:-1] / dy
upper_diag = mu_eff[1:] / dy
main_diag = -(np.concatenate(([0], lower_diag)) + np.concatenate((upper_diag, [0]))) / np.concatenate(([1], dy))

# Boundary conditions
main_diag[0] = 1.0
main_diag[-1] = 1.0

# Right-hand side
rhs = np.full(n, -1.0)
rhs[0] = 0.0  # ubar = 0 at y = 0
rhs[-1] = 0.0  # ubar = 0 at y = H

# Solve the linear system
A = diags([lower_diag, main_diag, upper_diag], offsets=[-1, 0, 1], format='csc')
ubar = spsolve(A, rhs)

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/ubar_Fully_Developed_Turbulent_Channel_Flow_CESS.npy', ubar)
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
H = 2.0  # Height of the domain
n = 100  # Number of grid points
mu = 1.0  # Molecular viscosity
kappa = 0.41  # von Karman constant
Re_tau = 180.0  # Friction Reynolds number
A = 26.0  # Constant in the Cess model

# Create a non-uniform grid clustered near the walls
y = np.linspace(0, H, n)
dy = np.diff(y)

# Compute y+ for the Cess model
y_plus = Re_tau * y / H

# Compute effective viscosity using the Cess model
mu_eff = np.zeros(n)
for i in range(n):
    term1 = (1 + (1/9) * kappa**2 * Re_tau**2 * (2*y[i] - y[i]**2)**2 * (3 - 4*y[i] + 2*y[i]**2)**2)
    term2 = (1 - np.exp(-y_plus[i] / A))
    mu_eff[i] = mu * (0.5 * (term1 * term2)**0.5 - 0.5)

# Discretize the PDE using finite differences
# Construct the matrix A and vector b for the linear system A u = b
diagonals = np.zeros((3, n))
b = np.full(n, -1.0)  # Right-hand side

# Fill the diagonals
for i in range(1, n-1):
    diagonals[0, i-1] = (mu_eff[i] + mu_eff[i-1]) / (2 * dy[i-1]**2)  # Lower diagonal
    diagonals[1, i] = -(mu_eff[i+1] + 2*mu_eff[i] + mu_eff[i-1]) / (2 * dy[i-1]**2)  # Main diagonal
    diagonals[2, i] = (mu_eff[i+1] + mu_eff[i]) / (2 * dy[i-1]**2)  # Upper diagonal

# Apply Dirichlet boundary conditions
diagonals[1, 0] = 1.0
diagonals[1, -1] = 1.0
b[0] = 0.0
b[-1] = 0.0

# Create sparse matrix A
A = diags([diagonals[0, :-1], diagonals[1], diagonals[2, 1:]], offsets=[-1, 0, 1], format='csc')

# Solve the linear system
u = spsolve(A, b)

# Save the final solution as a .npy file
np.save('velocity_profile.npy', u)
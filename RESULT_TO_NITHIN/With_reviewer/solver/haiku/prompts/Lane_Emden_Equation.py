import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve

# Grid setup
N = 1000  # Number of grid points
r = np.linspace(0, 1, N)
dr = r[1] - r[0]

# Initial guess
n = 3.0
R0 = 5.0
f = R0**(2/(n-1)) * (1 - r**2)**2

# Setup sparse matrix for Laplacian operator in spherical coordinates
# Central difference for interior points
main_diag = -2.0 * np.ones(N)
upper_diag = np.ones(N-1)
lower_diag = np.ones(N-1)

# Include r-dependent terms for spherical Laplacian
for i in range(1, N-1):
    lower_diag[i-1] = 1 + dr/(2*r[i])
    upper_diag[i] = 1 - dr/(2*r[i])
    
# Boundary conditions
main_diag[0] = 1.0
main_diag[-1] = 1.0
upper_diag[0] = 0.0
lower_diag[-1] = 0.0

# Create sparse matrix
A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], shape=(N, N))
A = csc_matrix(A / (dr**2))

# Newton iteration
max_iter = 100
tol = 1e-8
for iter in range(max_iter):
    # Compute residual
    f_old = f.copy()
    
    # Interior points
    R = (A @ f)[1:-1] + f[1:-1]**n
    
    # Jacobian matrix
    J = A.copy()
    J_diag = J.diagonal().copy()
    J_diag[1:-1] += n * f[1:-1]**(n-1)
    J.setdiag(J_diag)
    
    # Solve for correction
    b = np.zeros_like(f)
    b[1:-1] = -R
    b[0] = 0.0  # Center regularity
    b[-1] = -f[-1]  # Outer boundary condition
    
    df = spsolve(J, b)
    f += df
    
    # Check convergence
    if np.max(np.abs(f - f_old)) < tol:
        break

# Save solution
np.save('/PDE_Benchmark/results/prediction/haiku/prompts/f_Lane_Emden_Equation.npy', f)
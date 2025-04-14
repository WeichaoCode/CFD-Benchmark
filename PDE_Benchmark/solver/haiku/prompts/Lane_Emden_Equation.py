import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Grid setup
N = 1000  # Number of grid points
r = np.linspace(0, 1, N)
dr = r[1] - r[0]

# Initial guess
R0 = 5
n = 3.0
f = R0**(2/(n-1)) * (1 - r**2)**2

# Setup sparse matrix for Laplacian operator in spherical coordinates
# Central difference for interior points
r_i = r[1:-1]
main_diag = -2*np.ones(N-2)
upper_diag = (1 + 1/r_i[:-1])*np.ones(N-3)
lower_diag = (1 - 1/r_i[1:])*np.ones(N-3)

# Create sparse matrix
A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], shape=(N-2, N-2))
A = A/(dr**2)

# Newton iteration
max_iter = 100
tol = 1e-10
for iter in range(max_iter):
    f_old = f.copy()
    
    # Interior points
    b = -f[1:-1]**n
    
    # Boundary conditions
    b[0] -= (1 - 1/r_i[0])*f[0]/(dr**2)  # Center regularity
    b[-1] -= (1 + 1/r_i[-1])*0/(dr**2)    # Outer boundary f=0
    
    # Jacobian contribution from nonlinear term
    J = diags([-n*f[1:-1]**(n-1)], [0])
    
    # Solve linear system
    df = spsolve(A - J, b)
    
    # Update solution
    f[1:-1] += df
    
    # Check convergence
    if np.max(np.abs(f - f_old)) < tol:
        break

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/f_Lane_Emden_Equation.npy', f)
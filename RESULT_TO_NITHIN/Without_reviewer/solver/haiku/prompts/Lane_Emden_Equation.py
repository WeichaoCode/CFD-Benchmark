import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Grid setup
N = 1000  # Number of grid points
dr = 1.0/N
r = np.linspace(0, 1, N+1)  # Include r=0
r = r[1:]  # Remove r=0 point since equation is singular there

# Initial guess
R0 = 5.0
n = 3.0
f = R0**(2/(n-1)) * (1 - r**2)**2

# Setup sparse matrix for Laplacian operator
# d2f/dr2 + (2/r)df/dr terms
main_diag = -2.0*np.ones(N)
upper_diag = np.ones(N-1)
lower_diag = np.ones(N-1)

# Add (2/r)df/dr contribution
for i in range(N):
    upper_diag[i-1] += 1.0/r[i] if i > 0 else 0
    lower_diag[i-1] += -1.0/r[i] if i < N-1 else 0
    
A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
A = A/(dr**2)

# Newton iteration
max_iter = 100
tol = 1e-10
for iter in range(max_iter):
    # Compute residual
    f_laplace = (np.roll(f,-1) - 2*f + np.roll(f,1))/(dr**2)
    f_laplace[0] = (f[1] - 2*f[0] + f[1])/(dr**2)  # Use symmetry at r=0
    f_laplace[-1] = (-2*f[-1])/(dr**2)  # Boundary condition f(1)=0
    
    f_r = (np.roll(f,-1) - np.roll(f,1))/(2*dr)
    f_r[0] = 0  # Symmetry at r=0
    f_r[-1] = -f[-2]/(2*dr)  # One-sided difference at r=1
    
    R = f_laplace + (2/r)*f_r + f**n
    
    # Compute Jacobian
    J = A.copy()
    J.setdiag(J.diagonal() + n*f**(n-1))
    
    # Solve for update
    df = spsolve(J, -R)
    
    # Update solution
    f += df
    
    # Check convergence
    if np.max(np.abs(df)) < tol:
        break

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/f_Lane_Emden_Equation.npy', f)
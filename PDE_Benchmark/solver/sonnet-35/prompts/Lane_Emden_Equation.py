import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def solve_lane_emden(Nr=200, n=3.0):
    # Domain setup
    r = np.linspace(0, 1, Nr)
    dr = r[1] - r[0]
    
    # Initial guess
    R0 = 5.0
    f = R0**((2.0)/(n-1)) * (1 - r**2)**2
    
    # Nonlinear iteration parameters
    max_iter = 500
    tol = 1e-8
    
    # Nonlinear iteration
    for iter in range(max_iter):
        # Construct matrix for linearized problem
        main_diag = np.zeros(Nr)
        lower_diag = np.zeros(Nr-1)
        upper_diag = np.zeros(Nr-1)
        
        main_diag[0] = 1.0  # Symmetry at center
        main_diag[1:-1] = -2.0 - (2.0/r[1:-1]) * (1.0/dr) - n * f[1:-1]**(n-1)
        main_diag[-1] = 1.0  # Boundary condition
        
        lower_diag[:-1] = 1.0 + (1.0/r[1:-1]) * (1.0/dr)
        upper_diag[1:] = 1.0 - (1.0/r[1:-1]) * (1.0/dr)
        
        # Construct sparse matrix in CSR format
        diagonals = [main_diag, lower_diag, upper_diag]
        offsets = [0, -1, 1]
        A = sp.diags(diagonals, offsets, shape=(Nr, Nr)).tocsr()
        
        # Right-hand side
        b = np.zeros(Nr)
        b[-1] = 0.0  # Boundary condition
        
        # Solve linearized system
        df = spla.spsolve(A, b)
        
        # Update solution
        f_new = f + df
        
        # Check convergence
        if np.max(np.abs(df)) < tol:
            break
        
        f = f_new
    
    # Save solution
    np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/f_Lane_Emden_Equation.npy', f)

# Run solver
solve_lane_emden()
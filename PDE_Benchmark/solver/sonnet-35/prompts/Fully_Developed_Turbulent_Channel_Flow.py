import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Problem parameters
Re_tau = 395
mu = 1 / Re_tau
rho = 1.0
ny = 200  # Number of grid points
y = np.linspace(0, 2, ny)
dy = y[1] - y[0]

# Algebraic turbulence model (simple mixing length model)
def compute_mu_t(y):
    kappa = 0.41  # von Karman constant
    y_plus = y * Re_tau
    l_mix = kappa * y * (1 - np.exp(-y_plus/26))
    mu_t = rho * l_mix**2 * np.abs(np.gradient(np.zeros_like(y), dy))
    return mu_t

# Construct matrix and solve
def solve_momentum_equation():
    # Compute effective viscosity
    mu_t = compute_mu_t(y)
    mu_eff = mu + mu_t

    # Create sparse matrix for discretization
    diag_center = np.zeros(ny)
    diag_lower = np.zeros(ny-1)
    diag_upper = np.zeros(ny-1)
    rhs = np.zeros(ny)

    # Interior points discretization
    for i in range(1, ny-1):
        mu_w = 0.5 * (mu_eff[i] + mu_eff[i-1])
        mu_e = 0.5 * (mu_eff[i] + mu_eff[i+1])
        
        diag_lower[i-1] = -mu_w / dy**2
        diag_upper[i] = -mu_e / dy**2
        diag_center[i] = mu_w/dy**2 + mu_e/dy**2
        rhs[i] = -1.0

    # Boundary conditions
    diag_center[0] = 1.0
    diag_center[-1] = 1.0
    rhs[0] = 0.0
    rhs[-1] = 0.0

    # Create sparse matrix in CSR format
    main_diag = sp.diags(diag_center).tocsr()
    lower_diag = sp.diags(diag_lower, -1).tocsr()
    upper_diag = sp.diags(diag_upper, 1).tocsr()
    A = main_diag + lower_diag + upper_diag

    # Solve linear system
    u = spla.spsolve(A, rhs)
    return u

# Solve and save results
u = solve_momentum_equation()
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/u_Fully_Developed_Turbulent_Channel_Flow.npy', u)
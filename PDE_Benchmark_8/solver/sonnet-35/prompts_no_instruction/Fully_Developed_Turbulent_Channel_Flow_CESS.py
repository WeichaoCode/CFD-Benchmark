import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

# Problem parameters
H = 2.0  # Domain height
n = 100  # Number of grid points
mu = 0.001  # Molecular viscosity
kappa = 0.41  # von Karman constant
C_mu = 0.09  # Model constant

# Create non-uniform grid clustered near walls
y = np.zeros(n)
for i in range(n):
    y[i] = H * ((np.sin(np.pi * (i - (n-1)/2) / (n-1))) + 1) / 2

# Compute grid spacing
dy = np.diff(y)

# Initialize velocity
u = np.zeros(n)

# Compute eddy viscosity using Cess model
def compute_eddy_viscosity(u, y, mu):
    # Compute du/dy using central difference
    dudy = np.zeros(n)
    dudy[1:-1] = (u[2:] - u[:-2]) / (y[2:] - y[:-2])
    
    # Wall-adjacent points using one-sided difference
    dudy[0] = (u[1] - u[0]) / (y[1] - y[0])
    dudy[-1] = (u[-1] - u[-2]) / (y[-1] - y[-2])
    
    # Mixing length model
    mixing_length = kappa * (H/2 - np.abs(y - H/2))
    rho = 1.0  # Density assumption
    mu_t = rho * mixing_length**2 * np.abs(dudy)
    return mu_t

# Assemble linear system
def assemble_system(u, y, mu, mu_t):
    # Effective viscosity
    mu_eff = mu + mu_t
    
    # Create diagonal matrices
    diag_main = np.zeros(n)
    diag_lower = np.zeros(n-1)
    diag_upper = np.zeros(n-1)
    rhs = np.zeros(n)
    
    # Interior points
    for i in range(1, n-1):
        # Compute viscosity at interfaces
        mu_w = 0.5 * (mu_eff[i] + mu_eff[i-1])
        mu_e = 0.5 * (mu_eff[i] + mu_eff[i+1])
        
        # Compute grid spacings
        dy_west = y[i] - y[i-1]
        dy_east = y[i+1] - y[i]
        
        # Coefficients
        a_w = mu_w / dy_west
        a_e = mu_e / dy_east
        
        diag_main[i] = -(a_w + a_e)
        diag_lower[i-1] = a_w
        diag_upper[i] = a_e
        rhs[i] = -1.0  # Source term
    
    # Boundary conditions
    diag_main[0] = 1.0
    diag_main[-1] = 1.0
    rhs[0] = 0.0
    rhs[-1] = 0.0
    
    # Assemble sparse matrix
    main_diag = np.diag(diag_main)
    lower_diag = np.diag(diag_lower, k=-1)
    upper_diag = np.diag(diag_upper, k=1)
    A = sparse.csr_matrix(main_diag + lower_diag + upper_diag)
    
    return A, rhs

# Solve system iteratively
max_iter = 100
tol = 1e-6

for iter in range(max_iter):
    # Compute eddy viscosity
    mu_t = compute_eddy_viscosity(u, y, mu)
    
    # Assemble and solve system
    A, b = assemble_system(u, y, mu, mu_t)
    u_new = spla.spsolve(A, b)
    
    # Check convergence
    if np.max(np.abs(u_new - u)) < tol:
        u = u_new
        break
    
    u = u_new

# Save solution
save_values = ['u']
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/u_Fully_Developed_Turbulent_Channel_Flow_CESS.npy', u)
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Problem parameters
H = 2.0  # Domain height
mu = 0.001  # Molecular viscosity
ny = 100  # Number of grid points

# Grid generation
y = np.linspace(0, H, ny)
dy = y[1] - y[0]

# Turbulence model (Cess model)
def compute_eddy_viscosity(du_dy):
    kappa = 0.41  # von Karman constant
    C_mu = 0.09   # Model constant
    mixing_length = kappa * (H - np.abs(y - H/2))
    mu_t = rho * C_mu * mixing_length**2 * np.abs(du_dy)
    return mu_t

# Setup linear system
def assemble_system(mu, mu_t):
    # Effective viscosity
    mu_eff = mu + mu_t
    
    # Coefficient matrix (central difference)
    diags_A = np.zeros((3, ny))
    diags_A[0, 2:] = mu_eff[2:] / dy**2  # Lower diagonal
    diags_A[1, 1:-1] = -2 * mu_eff[1:-1] / dy**2  # Main diagonal
    diags_A[2, :-2] = mu_eff[:-2] / dy**2  # Upper diagonal
    
    # Boundary conditions
    diags_A[1, 0] = 1.0  # First row
    diags_A[1, -1] = 1.0  # Last row
    
    # Right-hand side
    b = np.zeros(ny)
    b[1:-1] = -1.0  # Source term
    b[0] = 0.0  # Dirichlet BC at bottom wall
    b[-1] = 0.0  # Dirichlet BC at top wall
    
    # Create sparse matrix in CSR format
    A = sp.diags(diags_A, [-1, 0, 1], shape=(ny, ny)).tocsr()
    
    return A, b

# Iterative solution with turbulence model
def solve_turbulent_flow():
    # Initial guess
    u = np.zeros(ny)
    
    # Maximum iterations
    max_iter = 100
    tolerance = 1e-6
    
    for iter in range(max_iter):
        # Compute velocity gradient
        du_dy = np.gradient(u, dy)
        
        # Update eddy viscosity
        mu_t = compute_eddy_viscosity(du_dy)
        
        # Assemble linear system
        A, b = assemble_system(mu, mu_t)
        
        # Solve system
        u_new = spla.spsolve(A, b)
        
        # Check convergence
        if np.linalg.norm(u_new - u) < tolerance:
            break
        
        u = u_new
    
    return u

# Solve problem
rho = 1.0  # Density
u_solution = solve_turbulent_flow()

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/haiku/prompts/u_solution_Fully_Developed_Turbulent_Channel_Flow.npy', u_solution)
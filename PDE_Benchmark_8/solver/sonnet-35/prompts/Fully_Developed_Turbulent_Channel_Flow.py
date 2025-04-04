import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Problem parameters
H = 2.0  # Domain height
ny = 200  # Number of grid points
mu = 0.001  # Molecular viscosity
k = 0.4  # von Karman constant
rho = 1.0  # Density

# Grid generation
y = np.linspace(0, H, ny)
dy = y[1] - y[0]

# Turbulent eddy viscosity (Cess model)
def compute_eddy_viscosity(u_profile):
    du_dy = np.gradient(u_profile, dy)
    l_mix = k * H * (1 - np.abs(y/H - 0.5))
    mu_t = rho * l_mix**2 * np.abs(du_dy)
    return mu_t

# Discretization of diffusion term
def create_diffusion_matrix(mu_eff):
    # Central difference discretization
    main_diag = np.zeros(ny)
    upper_diag = np.zeros(ny-1)
    lower_diag = np.zeros(ny-1)
    
    for i in range(1, ny-1):
        mu_plus = 0.5 * (mu_eff[i] + mu_eff[i+1])
        mu_minus = 0.5 * (mu_eff[i] + mu_eff[i-1])
        
        main_diag[i] = -(mu_plus + mu_minus) / dy**2
        upper_diag[i-1] = mu_plus / dy**2
        lower_diag[i] = mu_minus / dy**2
    
    # Apply boundary conditions
    main_diag[0] = 1.0
    main_diag[-1] = 1.0
    
    # Create sparse matrix in CSR format
    diagonals = [main_diag, lower_diag, upper_diag]
    offsets = [0, -1, 1]
    A = sp.diags(diagonals, offsets, shape=(ny, ny)).tocsr()
    
    return A

# Solution of the problem
def solve_velocity_profile():
    # Initial guess
    u = np.zeros(ny)
    
    # Source term
    b = np.ones(ny)
    b[0] = 0  # Dirichlet BC
    b[-1] = 0  # Dirichlet BC
    
    # Iterative solution
    max_iter = 100
    for _ in range(max_iter):
        # Compute eddy viscosity
        mu_t = compute_eddy_viscosity(u)
        mu_eff = mu + mu_t
        
        # Create diffusion matrix
        A = create_diffusion_matrix(mu_eff)
        
        # Solve linear system
        u_new = spla.spsolve(A, b)
        
        # Check convergence
        if np.linalg.norm(u_new - u) < 1e-6:
            break
        
        u = u_new
    
    return u

# Solve and save results
u_solution = solve_velocity_profile()

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts/u_solution_Fully_Developed_Turbulent_Channel_Flow.npy', u_solution)
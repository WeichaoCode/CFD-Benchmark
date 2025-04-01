import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Problem parameters
H = 2.0  # Domain height
n = 100  # Number of grid points
rho = 1.0  # Density
mu = 1e-3  # Molecular viscosity

# Turbulence model constants
beta_star = 0.09
beta = 0.075
sigma_k = 2.0
sigma_omega = 2.0
a1 = 0.31
CD_kw = 1e-10  # Small non-zero cross-diffusion term

# Non-uniform grid generation (clustered near walls)
def generate_grid(n, H):
    y = np.zeros(n)
    beta_cluster = 1.2  # Grid clustering parameter
    for i in range(n):
        xi = (i / (n-1)) * 2 - 1
        y[i] = H * (np.tanh(beta_cluster * xi) / np.tanh(beta_cluster))
    return y

# Grid generation
y = generate_grid(n, H)
dy = np.diff(y)

# Initialization with small non-zero values
k = np.full(n, 1e-6)
omega = np.full(n, 1e-6)

# Diffusion matrix generation
def create_diffusion_matrix(n, dy, mu_eff):
    # Create diagonal matrices for sparse representation
    diag = np.zeros(n)
    lower_diag = np.zeros(n-1)
    upper_diag = np.zeros(n-1)
    
    # Interior points
    for i in range(1, n-1):
        # Compute coefficients using central differences
        diag[i] = -(mu_eff[i-1]/dy[i-1] + mu_eff[i]/dy[i])
        lower_diag[i-1] = mu_eff[i-1]/dy[i-1]
        upper_diag[i] = mu_eff[i]/dy[i]
    
    # Boundary conditions (Dirichlet-like)
    diag[0] = 1.0
    diag[-1] = 1.0
    
    # Create sparse matrix in CSR format
    diagonals = [diag, lower_diag, upper_diag]
    offsets = [0, -1, 1]
    A = sp.diags(diagonals, offsets, shape=(n, n)).tocsr()
    
    return A

# Solve coupled SST turbulence model
def solve_sst_model(k, omega, y, rho, mu):
    # Iteration parameters
    max_iter = 200
    tol = 1e-8
    
    for iter in range(max_iter):
        # Compute strain rate and blending functions 
        # (simplified implementation with safeguards)
        S = max(1e-10, 1.0)  # Avoid zero division
        F1 = 1.0  # Blending function
        F2 = 1.0  # Blending function
        
        # Safeguard against division by zero
        omega_safe = np.maximum(omega, 1e-10)
        
        # Update eddy viscosity with safeguards
        mu_t = rho * k * np.minimum(1/omega_safe, a1/(S * F2))
        mu_t = np.maximum(mu_t, 0)
        
        # Compute effective viscosities
        mu_k = mu + np.maximum(mu_t / sigma_k, 0)
        mu_omega = mu + np.maximum(mu_t / sigma_omega, 0)
        
        # Production terms (simplified)
        P_k = mu_t * S**2
        
        # Safeguard mu_t in source terms
        mu_t_safe = np.maximum(mu_t, 1e-10)
        
        # Diffusion matrices
        A_k = create_diffusion_matrix(n, dy, mu_k)
        A_omega = create_diffusion_matrix(n, dy, mu_omega)
        
        # Source terms with additional safeguards
        b_k = P_k - beta_star * rho * k * omega_safe
        b_omega = (rho * P_k / mu_t_safe - beta * omega_safe**2 + 
                   (1 - F1) * CD_kw * k * omega_safe)
        
        # Solve for k and omega
        k_new = spla.spsolve(A_k, b_k)
        omega_new = spla.spsolve(A_omega, b_omega)
        
        # Enforce non-negativity
        k_new = np.maximum(k_new, 1e-10)
        omega_new = np.maximum(omega_new, 1e-10)
        
        # Convergence check
        if (np.max(np.abs(k_new - k)) < tol and 
            np.max(np.abs(omega_new - omega)) < tol):
            break
        
        k = k_new
        omega = omega_new
    
    return k, omega

# Solve the problem
k_final, omega_final = solve_sst_model(k, omega, y, rho, mu)

# Save results
save_values = ['k', 'omega', 'y']
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/k_final_Fully_Developed_Turbulent_Channel_Flow_SST.npy', k_final)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/omega_final_Fully_Developed_Turbulent_Channel_Flow_SST.npy', omega_final)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/y_Fully_Developed_Turbulent_Channel_Flow_SST.npy', y)
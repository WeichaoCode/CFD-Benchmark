import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Problem Parameters
H = 2.0  # Domain height
n = 100  # Number of grid points
dy = H / (n - 1)  # Grid spacing

# Turbulence Model Constants
C_mu = 0.22
C_1 = 1.4
C_2 = 0.6
sigma_k = 1.0
sigma_eps = 1.3
T_t = 1.0
L = 1.0  # Characteristic length scale

# Physical Parameters
rho = 1.0  # Density
mu = 1e-3  # Molecular viscosity
T = 1.0    # Timescale

# Create non-uniform grid (clustered near walls)
y = np.linspace(0, H, n)

# Finite Difference Discretization
def solve_v2f_model():
    # Initial Conditions
    k = np.zeros(n)
    eps = np.zeros(n)
    v2 = np.zeros(n)
    f = np.zeros(n)

    # Boundary Conditions
    k[0] = k[-1] = 0
    eps[0] = eps[-1] = 0
    v2[0] = v2[-1] = 0
    f[0] = f[-1] = 0

    # Iterations for coupled solution
    for _ in range(100):
        # Compute turbulent viscosity
        mu_t = C_mu * rho * np.sqrt(np.maximum(eps, 1e-10) / (np.maximum(k, 1e-10))) * T_t
        
        # Solve for k equation
        A_k = sp.diags([-1/(dy**2), 2/(dy**2), -1/(dy**2)], 
                       offsets=[-1, 0, 1], shape=(n, n)).tocsc()
        b_k = np.zeros(n)
        k_new = spla.spsolve(A_k, b_k)
        
        # Solve for epsilon equation
        A_eps = sp.diags([-1/(dy**2), 2/(dy**2), -1/(dy**2)], 
                         offsets=[-1, 0, 1], shape=(n, n)).tocsc()
        b_eps = np.zeros(n)
        eps_new = spla.spsolve(A_eps, b_eps)
        
        # Solve for v2 equation
        A_v2 = sp.diags([-1/(dy**2), 2/(dy**2), -1/(dy**2)], 
                        offsets=[-1, 0, 1], shape=(n, n)).tocsc()
        b_v2 = rho * k_new * f - 6 * rho * v2 * eps_new / (np.maximum(k_new, 1e-10))
        v2_new = spla.spsolve(A_v2, b_v2)
        
        # Solve for f equation
        A_f = sp.diags([-(1/L**2 + 1/(dy**2)), 2/(dy**2), -1/(dy**2)], 
                       offsets=[-1, 0, 1], shape=(n, n)).tocsc()
        b_f = (1/T) * (C_1 * (6 - v2_new) - (2/3) * (C_1 - 1)) - C_2 * 0  # Pk assumed 0
        f_new = spla.spsolve(A_f, b_f)
        
        # Update variables
        k, eps, v2, f = k_new, eps_new, v2_new, f_new
    
    return k, eps, v2, f

# Solve and save results
k, eps, v2, f = solve_v2f_model()

# Save results
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/k_Fully_Developed_Turbulent_Channel_Flow_V2F.npy', k)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/eps_Fully_Developed_Turbulent_Channel_Flow_V2F.npy', eps)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/v2_Fully_Developed_Turbulent_Channel_Flow_V2F.npy', v2)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/sonnet-35/prompts_no_instruction/f_Fully_Developed_Turbulent_Channel_Flow_V2F.npy', f)
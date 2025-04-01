import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve

# Constants
H = 2.0
n = 100
dy = H / (n - 1)
y = np.linspace(0, H, n)

# Model constants
C_mu = 0.09
C_e1 = 1.44
C_e2 = 1.92
sigma_k = 1.0
sigma_epsilon = 1.3
C_1 = 1.4
C_2 = 0.3
L = 0.1
T = 1.0
T_t = 1.0
rho = 1.0
mu = 1.0

# Initial conditions
k = np.full(n, 1e-5)  # Avoid division by zero
epsilon = np.full(n, 1e-5)  # Avoid division by zero
v2 = np.zeros(n)
f = np.zeros(n)

# Non-uniform grid clustering near the walls
y = np.linspace(0, 1, n)
y = H * (np.sinh(3 * y) / np.sinh(3))

# Helper functions
def compute_mu_t(k, epsilon):
    return C_mu * rho * np.sqrt(np.maximum(epsilon / np.maximum(k, 1e-10), 1e-10)) * T_t

def compute_Pk(v2, epsilon, k):
    return rho * k * f - 6 * rho * v2 * epsilon / np.maximum(k, 1e-10)

# Discretization matrices
def build_matrix(mu_t, sigma):
    diag_main = np.zeros(n)
    diag_upper = np.zeros(n - 1)
    diag_lower = np.zeros(n - 1)
    
    for i in range(1, n - 1):
        diag_main[i] = -(mu + mu_t[i] / sigma) / dy**2
        diag_upper[i] = (mu + mu_t[i] / sigma) / (2 * dy**2)
        diag_lower[i - 1] = (mu + mu_t[i] / sigma) / (2 * dy**2)
    
    diag_main[0] = diag_main[-1] = 1.0  # Dirichlet BCs
    return csc_matrix(diags([diag_main, diag_upper, diag_lower], [0, 1, -1]))

# Iterative solver
tolerance = 1e-6
max_iterations = 1000
for iteration in range(max_iterations):
    mu_t = compute_mu_t(k, epsilon)
    Pk = compute_Pk(v2, epsilon, k)
    
    # Solve for k
    A_k = build_matrix(mu_t, sigma_k)
    b_k = np.zeros(n)
    b_k[1:-1] = Pk[1:-1] - rho * epsilon[1:-1]
    k_new = spsolve(A_k, b_k)
    
    # Solve for epsilon
    A_epsilon = build_matrix(mu_t, sigma_epsilon)
    b_epsilon = np.zeros(n)
    b_epsilon[1:-1] = (1 / T) * (C_e1 * Pk[1:-1] - C_e2 * rho * epsilon[1:-1])
    epsilon_new = spsolve(A_epsilon, b_epsilon)
    
    # Solve for v2
    A_v2 = build_matrix(mu_t, sigma_k)
    b_v2 = np.zeros(n)
    b_v2[1:-1] = rho * k[1:-1] * f[1:-1] - 6 * rho * v2[1:-1] * epsilon[1:-1] / np.maximum(k[1:-1], 1e-10)
    v2_new = spsolve(A_v2, b_v2)
    
    # Solve for f
    A_f = build_matrix(mu_t, sigma_k)
    b_f = np.zeros(n)
    b_f[1:-1] = (1 / T) * (C_1 * (6 - v2[1:-1]) - (2 / 3) * (C_1 - 1)) - C_2 * Pk[1:-1]
    f_new = spsolve(A_f, b_f)
    
    # Check for convergence
    if (np.linalg.norm(k_new - k) < tolerance and
        np.linalg.norm(epsilon_new - epsilon) < tolerance and
        np.linalg.norm(v2_new - v2) < tolerance and
        np.linalg.norm(f_new - f) < tolerance):
        break
    
    k, epsilon, v2, f = k_new, epsilon_new, v2_new, f_new

    # Ensure non-negative values to prevent overflow
    k = np.maximum(k, 1e-10)
    epsilon = np.maximum(epsilon, 1e-10)
    v2 = np.maximum(v2, 0)
    f = np.maximum(f, 0)

# Save the final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/k_Fully_Developed_Turbulent_Channel_Flow_V2F.npy', k)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/epsilon_Fully_Developed_Turbulent_Channel_Flow_V2F.npy', epsilon)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/v2_Fully_Developed_Turbulent_Channel_Flow_V2F.npy', v2)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/f_Fully_Developed_Turbulent_Channel_Flow_V2F.npy', f)
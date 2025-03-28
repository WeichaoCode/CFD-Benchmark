import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

# Constants and parameters
H = 2.0
n = 100
rho = 1.0  # Density
mu = 1.0e-3  # Dynamic viscosity
beta_star = 0.09
beta = 0.075
a1 = 0.31
C_D = 0.0  # Assuming a constant for simplicity
F_1 = 1.0  # Assuming a constant for simplicity
F_2 = 1.0  # Assuming a constant for simplicity
P_k = 1.0  # Assuming a constant production term
S = 1.0  # Assuming a constant strain rate

# Non-uniform mesh clustering near the walls
y = np.linspace(0, H, n)
y = H * (np.sinh(3 * (y / H - 0.5)) / np.sinh(1.5) + 0.5)

# Initial conditions
k = np.zeros(n)
omega = np.zeros(n) + 1e-5  # Avoid division by zero by initializing omega with a small value

# Discretization parameters
dy = np.diff(y)
dy = np.append(dy, dy[-1])  # Append last element to match the size

# Helper function to compute mu_t
def compute_mu_t(k, omega):
    omega_safe = np.maximum(omega, 1e-5)  # Avoid division by zero
    return rho * k * np.minimum(1.0 / omega_safe, a1 / (S * F_2))

# Iterative solver for k and omega
def solve_k_omega(k, omega):
    mu_t = compute_mu_t(k, omega)
    
    # Discretize the equations for k
    A_k = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)).toarray()
    b_k = np.zeros(n)
    for i in range(1, n-1):
        A_k[i, i-1] = (mu + mu_t[i-1] / beta_star) / dy[i-1]**2
        A_k[i, i] = -(2 * (mu + mu_t[i] / beta_star) / dy[i]**2 + beta_star * rho * omega[i])
        A_k[i, i+1] = (mu + mu_t[i+1] / beta_star) / dy[i+1]**2
        b_k[i] = P_k
    
    # Boundary conditions for k
    A_k[0, 0] = A_k[-1, -1] = 1.0
    b_k[0] = b_k[-1] = 0.0
    
    # Convert A_k to CSR format
    A_k = csr_matrix(A_k)
    
    # Solve for k
    k = spsolve(A_k, b_k)
    
    # Discretize the equations for omega
    A_omega = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n)).toarray()
    b_omega = np.zeros(n)
    for i in range(1, n-1):
        A_omega[i, i-1] = (mu + mu_t[i-1] * omega[i-1]) / dy[i-1]**2
        A_omega[i, i] = -(2 * (mu + mu_t[i] * omega[i]) / dy[i]**2 + beta * omega[i]**2)
        A_omega[i, i+1] = (mu + mu_t[i+1] * omega[i+1]) / dy[i+1]**2
        b_omega[i] = rho * P_k / max(mu_t[i], 1e-5) + (1 - F_1) * C_D * k[i] * omega[i]
    
    # Boundary conditions for omega
    A_omega[0, 0] = A_omega[-1, -1] = 1.0
    b_omega[0] = b_omega[-1] = 0.0
    
    # Convert A_omega to CSR format
    A_omega = csr_matrix(A_omega)
    
    # Solve for omega
    omega = spsolve(A_omega, b_omega)
    
    return k, omega

# Solve the system
k, omega = solve_k_omega(k, omega)

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/k_Fully_Developed_Turbulent_Channel_Flow_SST.npy', k)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/omega_Fully_Developed_Turbulent_Channel_Flow_SST.npy', omega)
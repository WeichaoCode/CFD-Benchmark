import numpy as np

# Constants and parameters
H = 2.0
n = 100
beta_star = 0.09
beta = 0.075
sigma_k = 0.5
a1 = 0.31
C_D = 0.0  # Assuming a constant value for demonstration
rho = 1.0  # Assuming constant density
mu = 1.0e-5  # Dynamic viscosity
P_k = 1.0  # Assuming a constant production term for demonstration
F_1 = 1.0  # Assuming a constant blending function for demonstration
F_2 = 1.0  # Assuming a constant blending function for demonstration
S = 1.0  # Assuming a constant strain rate for demonstration

# Non-uniform grid clustering near the walls
y = np.linspace(0, H, n)
dy = np.gradient(y)

# Initial conditions
k = np.full(n, 1e-5)  # Small non-zero initial value to avoid division by zero
omega = np.full(n, 1e-5)  # Small non-zero initial value to avoid division by zero

# Finite Difference Method setup
def compute_mu_t(k, omega):
    return rho * k * np.minimum(1.0 / np.maximum(omega, 1e-10), a1 / (S * F_2))

def solve_k_omega(k, omega):
    mu_t = compute_mu_t(k, omega)
    for i in range(1, n-1):
        # Discretize the k-equation
        A_k = (mu + mu_t[i] / sigma_k) / dy[i]**2
        B_k = (mu + mu_t[i-1] / sigma_k) / dy[i-1]**2
        C_k = beta_star * rho * k[i] * omega[i] - P_k
        k[i] = (A_k * k[i+1] + B_k * k[i-1] - C_k) / (A_k + B_k)

        # Discretize the omega-equation
        A_omega = (mu + mu_t[i] * omega[i]) / dy[i]**2
        B_omega = (mu + mu_t[i-1] * omega[i-1]) / dy[i-1]**2
        C_omega = beta * omega[i]**2 - rho * P_k / np.maximum(mu_t[i], 1e-10) - (1 - F_1) * C_D * k[i] * omega[i]
        omega[i] = (A_omega * omega[i+1] + B_omega * omega[i-1] - C_omega) / (A_omega + B_omega)

# Iterative solver
tolerance = 1e-6
max_iterations = 10000
for iteration in range(max_iterations):
    k_old = k.copy()
    omega_old = omega.copy()
    solve_k_omega(k, omega)
    if np.linalg.norm(k - k_old) < tolerance and np.linalg.norm(omega - omega_old) < tolerance:
        break

# Save the final solutions
save_values = ['k', 'omega']
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/k_Fully_Developed_Turbulent_Channel_Flow_SST.npy', k)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/omega_Fully_Developed_Turbulent_Channel_Flow_SST.npy', omega)
import numpy as np

# Constants and parameters
H = 2.0
n = 100
beta_star = 0.09
beta = 0.075
a1 = 0.31
sigma_k = 0.5
C_D = 0.0  # Assuming a constant value for demonstration
rho = 1.0  # Assuming constant density
mu = 1.0e-5  # Dynamic viscosity
P_k = 1.0  # Assuming a constant production term for demonstration
F_1 = 1.0  # Assuming a constant blending function for demonstration
F_2 = 1.0  # Assuming a constant blending function for demonstration
S = 1.0  # Assuming a constant strain rate for demonstration
epsilon = 1e-10  # Small value to prevent division by zero

# Non-uniform grid clustering near the walls
y = np.linspace(0, H, n)
dy = np.gradient(y)

# Initial conditions
k = np.zeros(n)
omega = np.zeros(n)

# Finite Difference Method (FDM) setup
def compute_mu_t(k, omega):
    return rho * k * min(1.0 / (omega + epsilon), a1 / (S * F_2 + epsilon))

def solve_turbulence_model():
    global k, omega
    for i in range(1, n-1):
        mu_t = compute_mu_t(k[i], omega[i])
        
        # Discretize the k-equation
        A_k = (mu + mu_t / sigma_k) / dy[i]**2
        B_k = -beta_star * rho * omega[i]
        C_k = P_k
        
        k[i] = (A_k * (k[i+1] - 2*k[i] + k[i-1]) + C_k) / (1 - B_k)
        
        # Discretize the omega-equation
        A_omega = (mu + mu_t * omega[i]) / dy[i]**2
        B_omega = -beta * omega[i]**2
        C_omega = rho * P_k / (mu_t + epsilon) + (1 - F_1) * C_D * k[i] * omega[i]
        
        omega[i] = (A_omega * (omega[i+1] - 2*omega[i] + omega[i-1]) + C_omega) / (1 - B_omega)

# Solve the system
solve_turbulence_model()

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/k_Fully_Developed_Turbulent_Channel_Flow_SST.npy', k)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/omega_Fully_Developed_Turbulent_Channel_Flow_SST.npy', omega)
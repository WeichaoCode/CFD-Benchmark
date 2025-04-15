import numpy as np

# Constants
H = 2.0
n = 100
C_e1 = 1.44
C_e2 = 1.92
C_mu = 0.09
sigma_k = 1.0
sigma_epsilon = 1.3
rho = 1.0
mu = 1.0

# Non-uniform grid clustering near the walls
y = np.linspace(0, 1, n)
y = H * (np.sinh(3 * y) / np.sinh(3))

# Initial conditions
k = np.ones(n) * 1e-5  # Small non-zero initial value to avoid division by zero
epsilon = np.ones(n) * 1e-5  # Small non-zero initial value to avoid division by zero

# Functions for near-wall effects
def f_1(y):
    return 1.0

def f_2(y):
    return 1.0

def f_mu(y):
    return 1.0

# Discretization parameters
dy = np.diff(y)
dy = np.append(dy, dy[-1])  # To handle the last point

# Iterative solver parameters
tolerance = 1e-6
max_iterations = 10000

# Solver loop
for iteration in range(max_iterations):
    k_old = k.copy()
    epsilon_old = epsilon.copy()

    # Calculate turbulent viscosity
    mu_t = C_mu * f_mu(y) * rho * k**2 / epsilon

    # Solve for k
    for i in range(1, n-1):
        P_k = mu_t[i] * ((k[i+1] - k[i-1]) / (2 * dy[i]))**2
        A = (mu + mu_t[i] / sigma_k) / dy[i]**2
        B = (mu + mu_t[i] / sigma_k) / dy[i]**2
        C = -2 * (mu + mu_t[i] / sigma_k) / dy[i]**2 - rho * epsilon[i]
        k[i] = (A * k[i-1] + B * k[i+1] - P_k) / C

    # Solve for epsilon
    for i in range(1, n-1):
        A = (mu + mu_t[i] / sigma_epsilon) / dy[i]**2
        B = (mu + mu_t[i] / sigma_epsilon) / dy[i]**2
        C = -2 * (mu + mu_t[i] / sigma_epsilon) / dy[i]**2 - C_e2 * f_2(y[i]) * epsilon[i] / k[i]
        epsilon[i] = (A * epsilon[i-1] + B * epsilon[i+1] + C_e1 * f_1(y[i]) * P_k * epsilon[i] / k[i]) / C

    # Check for convergence
    if np.linalg.norm(k - k_old, ord=np.inf) < tolerance and np.linalg.norm(epsilon - epsilon_old, ord=np.inf) < tolerance:
        break

# Save the final solutions
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/k_Fully_Developed_Turbulent_Channel_Flow_KE.npy', k)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/epsilon_Fully_Developed_Turbulent_Channel_Flow_KE.npy', epsilon)
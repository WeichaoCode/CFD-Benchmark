import numpy as np

# Constants
C_e1 = 1.44
C_e2 = 1.92
C_mu = 0.09
sigma_k = 1.0
sigma_epsilon = 1.3
rho = 1.0  # Density
mu = 1.0e-5  # Dynamic viscosity

# Domain and grid
H = 2.0
n = 100
y = np.linspace(0, H, n)
dy = np.diff(y)
dy = np.append(dy, dy[-1])  # To handle the last point

# Initial conditions
k = np.zeros(n)
epsilon = np.full(n, 1e-6)  # Initialize epsilon to a small non-zero value

# Functions for near-wall effects (placeholders)
def f_1(y):
    return 1.0

def f_2(y):
    return 1.0

def f_mu(y):
    return 1.0

# Turbulent production term (placeholder)
def P_k(y):
    return 0.0

# Iterative solver
tolerance = 1e-6
max_iterations = 10000
converged = False

for iteration in range(max_iterations):
    k_old = k.copy()
    epsilon_old = epsilon.copy()
    
    # Update mu_t
    mu_t = C_mu * f_mu(y) * rho * k**2 / epsilon
    
    # Discretize and solve for k
    for i in range(1, n-1):
        A = (mu + mu_t[i] / sigma_k) / dy[i]
        B = (mu + mu_t[i-1] / sigma_k) / dy[i-1]
        C = A + B
        D = P_k(y[i]) - rho * epsilon[i]
        k[i] = (D + A * k[i+1] + B * k[i-1]) / C
    
    # Discretize and solve for epsilon
    for i in range(1, n-1):
        A = (mu + mu_t[i] / sigma_epsilon) / dy[i]
        B = (mu + mu_t[i-1] / sigma_epsilon) / dy[i-1]
        C = A + B
        D = (epsilon[i] / k[i]) * (C_e1 * f_1(y[i]) * P_k(y[i]) - C_e2 * f_2(y[i]) * epsilon[i])
        epsilon[i] = (D + A * epsilon[i+1] + B * epsilon[i-1]) / C
    
    # Check for convergence
    if np.linalg.norm(k - k_old, ord=np.inf) < tolerance and np.linalg.norm(epsilon - epsilon_old, ord=np.inf) < tolerance:
        converged = True
        break

if not converged:
    print("Warning: Solution did not converge within the maximum number of iterations.")

# Save the final solution
solution = np.vstack((k, epsilon)).T
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/solution_Fully_Developed_Turbulent_Channel_Flow_KE.npy', solution)
import numpy as np

# Constants
H = 2.0
n = 100
rho = 1.0  # Density
mu = 1.0e-3  # Dynamic viscosity
C_mu = 0.09
C_e1 = 1.44
C_e2 = 1.92
sigma_k = 1.0
sigma_epsilon = 1.3

# Discretization
y = np.linspace(0, H, n)
dy = np.diff(y)
dy = np.append(dy, dy[-1])  # To handle the last point

# Initial conditions with small positive values
k = np.full(n, 1e-6)
epsilon = np.full(n, 1e-6)

# Functions for near-wall effects (placeholders, should be defined based on the model)
def f1(y): return 1.0
def f2(y): return 1.0
def f_mu(y): return 1.0

# Small constant to prevent division by zero
epsilon_min = 1e-10

# Iterative solver parameters
tolerance = 1e-6
max_iterations = 10000

# Solver loop
for iteration in range(max_iterations):
    # Compute turbulent viscosity
    mu_t = C_mu * f_mu(y) * rho * k**2 / (epsilon + epsilon_min)

    # Discretize the equations
    A_k = np.zeros((n, n))
    b_k = np.zeros(n)
    A_epsilon = np.zeros((n, n))
    b_epsilon = np.zeros(n)

    for i in range(1, n-1):
        # Coefficients for k-equation
        a_w = (mu + mu_t[i-1] / sigma_k) / dy[i-1]
        a_e = (mu + mu_t[i] / sigma_k) / dy[i]
        a_p = a_w + a_e
        A_k[i, i-1] = -a_w
        A_k[i, i] = a_p
        A_k[i, i+1] = -a_e
        b_k[i] = rho * epsilon[i]

        # Coefficients for epsilon-equation
        a_w = (mu + mu_t[i-1] / sigma_epsilon) / dy[i-1]
        a_e = (mu + mu_t[i] / sigma_epsilon) / dy[i]
        a_p = a_w + a_e
        A_epsilon[i, i-1] = -a_w
        A_epsilon[i, i] = a_p
        A_epsilon[i, i+1] = -a_e
        b_epsilon[i] = (C_e1 * f1(y[i]) * rho * k[i] * epsilon[i] / (k[i] + epsilon_min) - C_e2 * f2(y[i]) * rho * epsilon[i])

    # Boundary conditions
    A_k[0, 0] = A_k[-1, -1] = 1.0
    A_epsilon[0, 0] = A_epsilon[-1, -1] = 1.0
    b_k[0] = b_k[-1] = 0.0
    b_epsilon[0] = b_epsilon[-1] = 0.0

    # Solve the linear systems
    k_new = np.linalg.solve(A_k, b_k)
    epsilon_new = np.linalg.solve(A_epsilon, b_epsilon)

    # Check for convergence
    if np.linalg.norm(k_new - k, ord=np.inf) < tolerance and np.linalg.norm(epsilon_new - epsilon, ord=np.inf) < tolerance:
        print(f"Converged in {iteration} iterations.")
        break

    k = k_new
    epsilon = epsilon_new

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_1/k_Fully_Developed_Turbulent_Channel_Flow_KE.npy', k)
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_1/epsilon_Fully_Developed_Turbulent_Channel_Flow_KE.npy', epsilon)
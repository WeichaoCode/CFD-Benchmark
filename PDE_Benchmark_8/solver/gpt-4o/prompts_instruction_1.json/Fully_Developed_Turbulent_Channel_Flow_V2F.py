import numpy as np

# Constants and parameters
H = 2.0  # Domain height
n = 100  # Number of grid points
dy = H / (n - 1)  # Grid spacing
rho = 1.0  # Density
C_mu = 0.09  # Model constant
sigma_k = 1.0  # Model constant
sigma_epsilon = 1.3  # Model constant
C_e1 = 1.44  # Model constant
C_e2 = 1.92  # Model constant
C_1 = 1.4  # Model constant
C_2 = 0.3  # Model constant
T = 1.0  # Time scale
L = 1.0  # Characteristic length scale
T_t = 1.0  # Turbulent temperature

# Initial conditions
k = np.full(n, 1e-6)  # Small positive value to avoid division by zero
epsilon = np.full(n, 1e-6)  # Small positive value to avoid division by zero
v2 = np.zeros(n)
f = np.zeros(n)

# Discretization matrices
A_k = np.zeros((n, n))
A_epsilon = np.zeros((n, n))
A_v2 = np.zeros((n, n))
A_f = np.zeros((n, n))

# Right-hand side vectors
b_k = np.zeros(n)
b_epsilon = np.zeros(n)
b_v2 = np.zeros(n)
b_f = np.zeros(n)

# Discretize the equations using central differences
for i in range(1, n-1):
    mu_t = C_mu * rho * (epsilon[i] / k[i])**0.5 * T_t
    A_k[i, i-1] = (mu_t / sigma_k + 1) / dy**2
    A_k[i, i] = -2 * (mu_t / sigma_k + 1) / dy**2
    A_k[i, i+1] = (mu_t / sigma_k + 1) / dy**2
    b_k[i] = rho * epsilon[i]

    A_epsilon[i, i-1] = (mu_t / sigma_epsilon + 1) / dy**2
    A_epsilon[i, i] = -2 * (mu_t / sigma_epsilon + 1) / dy**2
    A_epsilon[i, i+1] = (mu_t / sigma_epsilon + 1) / dy**2
    b_epsilon[i] = (1 / T) * (C_e1 * rho * epsilon[i] - C_e2 * rho * epsilon[i])

    A_v2[i, i-1] = (mu_t / sigma_k + 1) / dy**2
    A_v2[i, i] = -2 * (mu_t / sigma_k + 1) / dy**2
    A_v2[i, i+1] = (mu_t / sigma_k + 1) / dy**2
    b_v2[i] = rho * k[i] * f[i] - 6 * rho * v2[i] * epsilon[i] / k[i]

    A_f[i, i-1] = L**2 / dy**2
    A_f[i, i] = -2 * L**2 / dy**2 - 1
    A_f[i, i+1] = L**2 / dy**2
    b_f[i] = (1 / T) * (C_1 * (6 - v2[i]) - (2 / 3) * (C_1 - 1)) - C_2 * rho * epsilon[i]

# Boundary conditions
A_k[0, 0] = A_k[-1, -1] = 1
A_epsilon[0, 0] = A_epsilon[-1, -1] = 1
A_v2[0, 0] = A_v2[-1, -1] = 1
A_f[0, 0] = A_f[-1, -1] = 1

# Solve the linear systems
k = np.linalg.solve(A_k, b_k)
epsilon = np.linalg.solve(A_epsilon, b_epsilon)
v2 = np.linalg.solve(A_v2, b_v2)
f = np.linalg.solve(A_f, b_f)

# Save the final solution
np.save('solution.npy', np.array([k, epsilon, v2, f]))
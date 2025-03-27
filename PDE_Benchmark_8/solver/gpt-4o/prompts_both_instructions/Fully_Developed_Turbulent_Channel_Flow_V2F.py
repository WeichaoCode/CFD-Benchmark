import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

# Constants and parameters
H = 2.0  # Domain height
n = 100  # Number of grid points
C_mu = 0.09
C_e1 = 1.44
C_e2 = 1.92
sigma_k = 1.0
sigma_epsilon = 1.3
C1 = 1.4
C2 = 0.3
L = 0.1
rho = 1.0  # Density
T_t = 1.0  # Turbulent temperature

# Create a non-uniform mesh clustered near the walls
y = np.linspace(0, H, n)
dy = np.gradient(y)

# Function to compute turbulent viscosity
def compute_mu_t(k, epsilon):
    return C_mu * rho * (epsilon / k)**0.5 * T_t if k > 0 else 0

# Discretize and solve the equations
def solve_v2f():
    # Initialize fields
    k = np.zeros(n)
    epsilon = np.zeros(n)
    v2 = np.zeros(n)
    f = np.zeros(n)

    # Initialize matrices and vectors
    A_k = np.zeros((3, n))
    b_k = np.zeros(n)
    A_epsilon = np.zeros((3, n))
    b_epsilon = np.zeros(n)
    A_v2 = np.zeros((3, n))
    b_v2 = np.zeros(n)
    A_f = np.zeros((3, n))
    b_f = np.zeros(n)

    # Fill matrices and vectors for k, epsilon, v2, and f
    for i in range(1, n-1):
        mu_t = compute_mu_t(k[i], epsilon[i])
        
        # Discretize k-equation
        A_k[0, i] = (mu_t / sigma_k + mu_t) / dy[i]**2
        A_k[1, i] = -2 * (mu_t / sigma_k + mu_t) / dy[i]**2
        A_k[2, i] = (mu_t / sigma_k + mu_t) / dy[i]**2
        b_k[i] = rho * epsilon[i]

        # Discretize epsilon-equation
        A_epsilon[0, i] = (mu_t / sigma_epsilon + mu_t) / dy[i]**2
        A_epsilon[1, i] = -2 * (mu_t / sigma_epsilon + mu_t) / dy[i]**2
        A_epsilon[2, i] = (mu_t / sigma_epsilon + mu_t) / dy[i]**2
        b_epsilon[i] = C_e1 * rho * epsilon[i] - C_e2 * rho * epsilon[i]

        # Discretize v2-equation
        A_v2[0, i] = (mu_t / sigma_k + mu_t) / dy[i]**2
        A_v2[1, i] = -2 * (mu_t / sigma_k + mu_t) / dy[i]**2
        A_v2[2, i] = (mu_t / sigma_k + mu_t) / dy[i]**2
        b_v2[i] = rho * k[i] * f[i] - 6 * rho * v2[i] * epsilon[i] / k[i] if k[i] > 0 else 0

        # Discretize f-equation
        A_f[0, i] = L**2 / dy[i]**2
        A_f[1, i] = -2 * L**2 / dy[i]**2 - 1
        A_f[2, i] = L**2 / dy[i]**2
        b_f[i] = C1 * (6 - v2[i]) - 2/3 * (C1 - 1) - C2 * rho * epsilon[i]

    # Apply boundary conditions
    A_k[1, 0] = A_k[1, -1] = 1
    A_epsilon[1, 0] = A_epsilon[1, -1] = 1
    A_v2[1, 0] = A_v2[1, -1] = 1
    A_f[1, 0] = A_f[1, -1] = 1

    # Construct sparse matrices using correct diagonal sizes
    diagonals_k = [A_k[0, 1:], A_k[1, :], A_k[2, :-1]]
    diagonals_epsilon = [A_epsilon[0, 1:], A_epsilon[1, :], A_epsilon[2, :-1]]
    diagonals_v2 = [A_v2[0, 1:], A_v2[1, :], A_v2[2, :-1]]
    diagonals_f = [A_f[0, 1:], A_f[1, :], A_f[2, :-1]]

    # Convert to CSR format
    A_k_csr = diags(diagonals_k, [-1, 0, 1], shape=(n, n)).tocsr()
    A_epsilon_csr = diags(diagonals_epsilon, [-1, 0, 1], shape=(n, n)).tocsr()
    A_v2_csr = diags(diagonals_v2, [-1, 0, 1], shape=(n, n)).tocsr()
    A_f_csr = diags(diagonals_f, [-1, 0, 1], shape=(n, n)).tocsr()

    # Solve the linear systems
    k = spsolve(A_k_csr, b_k)
    epsilon = spsolve(A_epsilon_csr, b_epsilon)
    v2 = spsolve(A_v2_csr, b_v2)
    f = spsolve(A_f_csr, b_f)

    return k, epsilon, v2, f

# Solve the system
k, epsilon, v2, f = solve_v2f()

# Save the final solution
np.save('solution.npy', np.array([k, epsilon, v2, f]))

# Plot the velocity profile
plt.plot(y, v2, label='Turbulent velocity profile')
plt.xlabel('y')
plt.ylabel('v^2')
plt.title('Turbulent Velocity Profile')
plt.legend()
plt.show()
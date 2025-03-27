import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
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
L = 1.0
T = 1.0
rho = 1.0
mu = 1.0

# Initialize the mesh with clustering near the walls
y = np.linspace(0, H, n)
dy = np.gradient(y)

# Initial conditions with small non-zero values
k = np.full(n, 1e-5)
epsilon = np.full(n, 1e-5)
v2 = np.zeros(n)
f = np.zeros(n)

# Discretize the governing equations
def compute_mu_t(k, epsilon):
    return C_mu * rho * (epsilon / k)**0.5 * T

def solve_pde():
    # Initialize arrays for the linear system
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Discretize the equations using finite differences
    for i in range(1, n-1):
        mu_t = compute_mu_t(k[i], epsilon[i])
        
        # Discretize the k-equation
        A[i, i-1] = (mu + mu_t / sigma_k) / dy[i]**2
        A[i, i] = -2 * (mu + mu_t / sigma_k) / dy[i]**2
        A[i, i+1] = (mu + mu_t / sigma_k) / dy[i]**2
        b[i] = rho * epsilon[i] - P_k(i)

        # Discretize the epsilon-equation
        A[i, i-1] += (mu + mu_t / sigma_epsilon) / dy[i]**2
        A[i, i] += -2 * (mu + mu_t / sigma_epsilon) / dy[i]**2
        A[i, i+1] += (mu + mu_t / sigma_epsilon) / dy[i]**2
        b[i] += (1/T) * (C_e1 * P_k(i) - C_e2 * rho * epsilon[i])

        # Discretize the v2-equation
        A[i, i-1] += (mu + mu_t / sigma_k) / dy[i]**2
        A[i, i] += -2 * (mu + mu_t / sigma_k) / dy[i]**2
        A[i, i+1] += (mu + mu_t / sigma_k) / dy[i]**2
        b[i] += rho * k[i] * f[i] - 6 * rho * v2[i] * epsilon[i] / k[i]

    # Apply boundary conditions
    A[0, 0] = A[-1, -1] = 1.0
    b[0] = b[-1] = 0.0

    # Solve the linear system
    u = np.linalg.solve(A, b)
    return u

def P_k(i):
    # Placeholder for turbulent production term
    return 0.0

# Solve the PDE
solution = solve_pde()

# Save the solution as a .npy file
np.save('solution.npy', solution)

# Plot the velocity profile
plt.plot(y, solution, label='Turbulent Velocity Profile')
plt.xlabel('y')
plt.ylabel('Velocity')
plt.title('Velocity Profile')
plt.legend()
plt.show()
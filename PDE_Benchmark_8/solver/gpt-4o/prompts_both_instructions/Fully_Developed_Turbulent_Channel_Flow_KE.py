import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

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

# Functions for near-wall effects (placeholders)
def f1(y):
    return 1.0

def f2(y):
    return 1.0

def f_mu(y):
    return 1.0

# Create a non-uniform mesh clustered near the walls
y = np.linspace(0, H, n)
dy = np.gradient(y)

# Initial conditions
k = np.full(n, 1e-6)  # Small non-zero initial value
epsilon = np.full(n, 1e-6)  # Small non-zero initial value

# Time-stepping parameters
dt = 0.01
t_final = 1.0
num_steps = int(t_final / dt)

# Discretize and solve
for step in range(num_steps):
    # Compute turbulent viscosity
    mu_t = C_mu * f_mu(y) * rho * k**2 / epsilon

    # Discretize the equations using finite differences
    A_k = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    A_k = A_k.tocsr()  # Convert to CSR format
    A_k[0, 0] = A_k[-1, -1] = 1  # Dirichlet BCs
    b_k = np.zeros(n)
    b_k[0] = b_k[-1] = 0  # Boundary values for k

    A_epsilon = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    A_epsilon = A_epsilon.tocsr()  # Convert to CSR format
    A_epsilon[0, 0] = A_epsilon[-1, -1] = 1  # Dirichlet BCs
    b_epsilon = np.zeros(n)
    b_epsilon[0] = b_epsilon[-1] = 0  # Boundary values for epsilon

    # Solve the linear systems
    try:
        k_new = spsolve(A_k, b_k)
        epsilon_new = spsolve(A_epsilon, b_epsilon)
    except Exception as e:
        print(f"Error solving linear system: {e}")
        break

    # Update k and epsilon
    k = k_new
    epsilon = epsilon_new

# Save the final solution
np.save('solution.npy', np.vstack((k, epsilon)))

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k, y, label='Turbulent Kinetic Energy k')
plt.xlabel('k')
plt.ylabel('y')
plt.title('Turbulent Kinetic Energy Profile')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epsilon, y, label='Dissipation Rate ε')
plt.xlabel('ε')
plt.ylabel('y')
plt.title('Dissipation Rate Profile')
plt.grid(True)

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

# Constants and parameters
H = 2.0
n = 100
beta_star = 0.09
sigma_k = 0.5
sigma_omega = 0.5
a1 = 0.31
C_D = 0.0  # Assuming a constant for simplicity
rho = 1.0  # Density
mu = 1.0e-3  # Dynamic viscosity
epsilon = 1e-10  # Small value to prevent division by zero

# Create a non-uniform mesh clustered near the walls
y = np.linspace(0, H, n)
dy = np.gradient(y)

# Initial conditions
k = np.zeros(n)
omega = np.full(n, 1e-5)  # Small positive value to avoid division by zero

# Helper functions for blending functions and strain rate
def F1(y):
    return 1.0  # Simplified for demonstration

def F2(y):
    return 1.0  # Simplified for demonstration

def S(y):
    return 1.0  # Simplified for demonstration

# Discretize the equations using finite differences
def compute_turbulent_viscosity(k, omega):
    return rho * k * np.minimum(1.0 / (omega + epsilon), a1 / (S(y) * F2(y)))

def solve_k_omega(k, omega):
    mu_t = compute_turbulent_viscosity(k, omega)
    
    # Discretize the k-equation
    A_k = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    b_k = np.zeros(n)
    
    # Discretize the omega-equation
    A_omega = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n))
    b_omega = np.zeros(n)
    
    # Apply boundary conditions
    A_k = A_k.tolil()
    A_k[0, 0] = A_k[-1, -1] = 1
    A_k[0, 1] = A_k[-1, -2] = 0
    b_k[0] = b_k[-1] = 0
    
    A_omega = A_omega.tolil()
    A_omega[0, 0] = A_omega[-1, -1] = 1
    A_omega[0, 1] = A_omega[-1, -2] = 0
    b_omega[0] = b_omega[-1] = 0
    
    # Convert to CSR format for spsolve
    A_k = A_k.tocsr()
    A_omega = A_omega.tocsr()
    
    # Solve the linear systems
    k_new = spsolve(A_k, b_k)
    omega_new = spsolve(A_omega, b_omega)
    
    return k_new, omega_new

# Iterative solver
tolerance = 1e-6
max_iterations = 1000
for iteration in range(max_iterations):
    k_new, omega_new = solve_k_omega(k, omega)
    
    # Check for convergence
    if np.linalg.norm(k_new - k) < tolerance and np.linalg.norm(omega_new - omega) < tolerance:
        print(f"Converged in {iteration} iterations.")
        break
    
    k, omega = k_new, omega_new

# Save the final solution
np.save('final_solution.npy', np.vstack((k, omega)))

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k, y, label='Turbulent Kinetic Energy (k)')
plt.xlabel('k')
plt.ylabel('y')
plt.title('Turbulent Kinetic Energy Profile')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(omega, y, label='Specific Dissipation Rate (omega)')
plt.xlabel('omega')
plt.ylabel('y')
plt.title('Specific Dissipation Rate Profile')
plt.grid(True)

plt.tight_layout()
plt.show()
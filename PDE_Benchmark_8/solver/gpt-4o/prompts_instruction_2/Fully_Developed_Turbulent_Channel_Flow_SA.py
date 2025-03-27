import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Constants
H = 2.0  # Height of the domain
n = 100  # Number of grid points
mu = 1.0e-3  # Molecular viscosity (example value)

# Create a non-uniform mesh clustered near the walls
y = np.linspace(0, H, n)
y = H * (np.sinh(3 * (y / H - 0.5)) / np.sinh(1.5) + 0.5)

# Initialize velocity and turbulent viscosity
u = np.zeros(n)
mu_t = np.zeros(n)

# Spalart-Allmaras model parameters (example values)
C_b1 = 0.1355
C_w2 = 0.3
sigma = 2/3
kappa = 0.41
C_v1 = 7.1
C_w3 = 2.0

# Compute turbulent viscosity using Spalart-Allmaras model
def compute_mu_t(y, u, mu, mu_t):
    # Example implementation of the Spalart-Allmaras model
    # This is a simplified version and may not be fully accurate
    for i in range(1, n-1):
        dy = y[i+1] - y[i]
        du_dy = (u[i+1] - u[i]) / dy
        mu_t[i] = C_v1 * mu * (du_dy / (C_w2 * (du_dy**2 + (mu / (kappa * dy))**2)))
    return mu_t

mu_t = compute_mu_t(y, u, mu, mu_t)

# Effective viscosity
mu_eff = mu + mu_t

# Discretize the PDE using finite differences
dy = np.diff(y)
A = diags([-mu_eff[1:-1]/dy[:-1], (mu_eff[1:-1]/dy[:-1] + mu_eff[2:]/dy[1:]), -mu_eff[2:]/dy[1:]], [-1, 0, 1], shape=(n-2, n-2)).tocsr()
b = -np.ones(n-2)

# Apply boundary conditions
u[0] = 0
u[-1] = 0

# Solve the linear system
u[1:-1] = spsolve(A, b)

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_2/u_Fully_Developed_Turbulent_Channel_Flow_SA.npy', u)

# Plot the velocity profile
plt.plot(u, y, label='Turbulent Velocity Profile')
plt.xlabel('Velocity')
plt.ylabel('y')
plt.title('Velocity Profile')
plt.legend()
plt.show()
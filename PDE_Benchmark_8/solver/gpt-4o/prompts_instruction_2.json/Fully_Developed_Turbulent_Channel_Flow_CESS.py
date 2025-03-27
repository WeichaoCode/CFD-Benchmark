import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Constants
H = 2.0  # Height of the domain
n = 100  # Number of grid points
nu = 1.0e-3  # Molecular viscosity (example value)

# Cess model parameters
A_plus = 26.0
B = 1.0 / 0.41
C = 5.5

# Create non-uniform mesh clustered near the walls
y = np.linspace(0, H, n)
y = H * (np.sinh(B * (y / H - 0.5)) / np.sinh(B / 2) + 0.5)
dy = np.diff(y)

# Initialize velocity and effective viscosity
u = np.zeros(n)
mu_eff = np.zeros(n)

# Compute effective viscosity using the Cess model
for i in range(n):
    y_plus = y[i] / nu
    mu_t = nu * (1.0 + (A_plus * y_plus) ** 2) ** 0.5 * (1.0 - np.exp(-y_plus / C)) ** 2
    mu_eff[i] = nu + mu_t

# Discretize the PDE using finite differences
A = np.zeros((n, n))
b = np.zeros(n)

# Fill the matrix A and vector b
for i in range(1, n - 1):
    A[i, i - 1] = mu_eff[i - 1] / dy[i - 1] ** 2
    A[i, i] = -(mu_eff[i - 1] / dy[i - 1] ** 2 + mu_eff[i] / dy[i] ** 2)
    A[i, i + 1] = mu_eff[i] / dy[i] ** 2
    b[i] = -1

# Apply Dirichlet boundary conditions
A[0, 0] = 1.0
A[-1, -1] = 1.0
b[0] = 0.0
b[-1] = 0.0

# Solve the linear system
u = np.linalg.solve(A, b)

# Save the final solution as a .npy file
np.save('velocity_profile.npy', u)

# Plot the velocity profile
plt.plot(u, y, label='Turbulent Profile')
plt.xlabel('Velocity')
plt.ylabel('y')
plt.title('Velocity Profile')
plt.legend()
plt.show()
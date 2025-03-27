import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Constants
H = 2.0  # Domain height
n = 100  # Number of grid points
mu = 1.0  # Molecular viscosity
kappa = 0.41  # von Karman constant
Re_tau = 1000  # Friction Reynolds number
A = 26  # Cess model constant

# Create non-uniform mesh clustered near the walls
y = np.linspace(0, H, n)
y = 0.5 * (1 - np.cos(np.pi * y / H)) * H  # Clustering using cosine transformation
dy = np.diff(y)

# Compute y+ for each grid point
y_plus = y * Re_tau / H

# Compute effective viscosity using the Cess turbulence model
mu_eff = np.zeros(n)
for i in range(n):
    term = (1 + (1/9) * kappa**2 * Re_tau**2 * (2*y[i] - y[i]**2)**2 * (3 - 4*y[i] + 2*y[i]**2)**2 *
            (1 - np.exp(-y_plus[i] / A)))
    mu_eff[i] = mu * (0.5 * np.sqrt(term) - 0.5)

# Discretize the governing equation using finite differences
# Construct the matrix A and vector b for the linear system A u = b
diagonals = np.zeros((3, n))
b = np.full(n, -1.0)  # Right-hand side

# Fill the diagonals
for i in range(1, n-1):
    diagonals[0, i] = mu_eff[i-1] / dy[i-1]**2  # Lower diagonal
    diagonals[1, i] = -(mu_eff[i-1] / dy[i-1]**2 + mu_eff[i] / dy[i]**2)  # Main diagonal
    diagonals[2, i] = mu_eff[i] / dy[i]**2  # Upper diagonal

# Apply boundary conditions
diagonals[1, 0] = 1.0
diagonals[1, -1] = 1.0
b[0] = 0.0
b[-1] = 0.0

# Adjust the diagonals to match the matrix size
diagonals[0, 0] = 0.0  # No lower diagonal at the first row
diagonals[2, -1] = 0.0  # No upper diagonal at the last row

# Create sparse matrix A
A = diags(diagonals, offsets=[-1, 0, 1], shape=(n, n), format='csc')

# Solve the linear system
u = spsolve(A, b)

# Save the final solution as a .npy file
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_Fully_Developed_Turbulent_Channel_Flow_CESS.npy', u)

# Plot the velocity profile
plt.plot(u, y, label='Turbulent Velocity Profile')
plt.xlabel('Velocity')
plt.ylabel('y')
plt.title('Velocity Profile')
plt.legend()
plt.grid()
plt.show()
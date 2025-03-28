import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Parameters
H = 2.0  # Height of the domain
n = 100  # Number of grid points
mu = 1.0  # Molecular viscosity
kappa = 0.41  # von Karman constant
Re_tau = 5200  # Friction Reynolds number
A = 26  # Constant in the Cess model

# Create a non-uniform mesh clustered near the walls
y = np.linspace(0, 1, n)
y = H * (1 - np.cos(np.pi * y)) / 2  # Clustering using cosine transformation
dy = np.diff(y)

# Initialize velocity
u = np.zeros(n)

# Compute effective viscosity using the Cess model
y_plus = Re_tau * y / H
mu_eff = mu * (0.5 * (1 + (1/9) * kappa**2 * Re_tau**2 * (2*y - y**2)**2 * (3 - 4*y + 2*y**2)**2 *
                      (1 - np.exp(-y_plus/A)))**0.5 - 0.5)

# Discretize the governing equation using finite differences
# Construct the matrix A and vector b for the linear system A u = b
diagonals = np.zeros((3, n))
b = np.full(n, -1.0)  # Right-hand side

# Apply boundary conditions
diagonals[1, 0] = 1.0
diagonals[1, -1] = 1.0
b[0] = 0.0
b[-1] = 0.0

# Fill the matrix A for the interior points
for i in range(1, n-1):
    dy1 = dy[i-1]
    dy2 = dy[i]
    mu_eff1 = mu_eff[i-1]
    mu_eff2 = mu_eff[i]
    
    diagonals[0, i] = (mu_eff1 + mu_eff2) / (2 * dy1)  # Lower diagonal
    diagonals[1, i] = -(mu_eff1 / dy1 + mu_eff2 / dy2)  # Main diagonal
    diagonals[2, i] = (mu_eff1 + mu_eff2) / (2 * dy2)  # Upper diagonal

# Create the sparse matrix in CSR format
A_matrix = diags(diagonals, offsets=[-1, 0, 1], shape=(n, n)).tocsr()

# Solve the linear system
u = spsolve(A_matrix, b)

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/u_Fully_Developed_Turbulent_Channel_Flow_CESS.npy', u)

# Plot the velocity profile
plt.plot(y, u, label='Turbulent Profile')
plt.xlabel('y')
plt.ylabel('Velocity $\overline{u}$')
plt.title('Velocity Profile')
plt.legend()
plt.show()
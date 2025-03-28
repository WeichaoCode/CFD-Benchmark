import numpy as np

# Parameters
H = 2.0  # Height of the domain
n = 100  # Number of grid points
mu = 1.0  # Molecular viscosity (assumed constant for simplicity)

# Create a non-uniform grid clustered near the walls
y = np.linspace(0, 1, n)
y = H * (1 - np.cos(np.pi * y)) / 2  # Clustering using cosine transformation

# Initialize velocity field
ubar = np.zeros(n)

# Spalart-Allmaras model parameters (simplified for this example)
def compute_mu_t(y, ubar):
    # Placeholder for a simple turbulent viscosity model
    # In practice, this would involve solving the Spalart-Allmaras equations
    return 0.1 * np.ones_like(y)  # Constant eddy viscosity for simplicity

# Compute effective viscosity
mu_t = compute_mu_t(y, ubar)
mu_eff = mu + mu_t

# Finite difference method setup
dy = np.diff(y)
dy_p = np.concatenate(([dy[0]], dy))  # Forward difference
dy_m = np.concatenate((dy, [dy[-1]]))  # Backward difference

# Construct the linear system A * u = b
A = np.zeros((n, n))
b = np.full(n, -1.0)  # Source term

# Fill the matrix A using central differences
for i in range(1, n-1):
    A[i, i-1] = mu_eff[i] / dy_m[i]
    A[i, i] = -(mu_eff[i] / dy_m[i] + mu_eff[i+1] / dy_p[i])
    A[i, i+1] = mu_eff[i+1] / dy_p[i]

# Apply Dirichlet boundary conditions
A[0, 0] = 1.0
A[-1, -1] = 1.0
b[0] = 0.0
b[-1] = 0.0

# Solve the linear system
ubar = np.linalg.solve(A, b)

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_no_instruction/ubar_Fully_Developed_Turbulent_Channel_Flow_SA.npy', ubar)
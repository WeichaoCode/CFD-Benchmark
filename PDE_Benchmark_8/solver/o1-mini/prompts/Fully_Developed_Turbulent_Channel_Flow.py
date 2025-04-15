import numpy as np

# Parameters
H = 2.0            # Domain height
N = 100            # Number of grid points
mu = 1.0           # Molecular viscosity
A = 1.0            # Cess model constant

# Discretize the domain
y = np.linspace(0, H, N)
dy = y[1] - y[0]

# Compute turbulent eddy viscosity using Cess model
mu_t = A * y * (1 - y / H)
mu_eff = mu + mu_t

# Compute mu_eff at half points
mu_eff_half = (mu_eff[:-1] + mu_eff[1:]) / 2.0

# Initialize the coefficient matrix and RHS vector
A_matrix = np.zeros((N-2, N-2))
b = -1.0 * np.ones(N-2)

# Fill the coefficient matrix
for i in range(N-2):
    if i > 0:
        A_matrix[i, i-1] = -mu_eff_half[i] / dy**2
    A_matrix[i, i] = (mu_eff_half[i] + mu_eff_half[i+1]) / dy**2
    if i < N-3:
        A_matrix[i, i+1] = -mu_eff_half[i+1] / dy**2

# Solve the linear system
u_interior = np.linalg.solve(A_matrix, b)

# Assemble the full solution with boundary conditions
u = np.zeros(N)
u[1:N-1] = u_interior

# Save the solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/o1-mini/prompts/u_Fully_Developed_Turbulent_Channel_Flow.npy', u)
#!/usr/bin/env python3
import numpy as np

# Problem parameters
L = 0.1                # Length of the domain in meters
phi_left = 10.0        # Dirichlet boundary condition at x=0
phi_right = 100.0      # Dirichlet boundary condition at x=0.1
Gamma = 1e-4           # Diffusion coefficient in m^2/s

# Discretization parameters
N = 51                 # Number of nodes (including boundaries)
dx = L / (N - 1)       # Uniform grid spacing

# Create grid (cell centers for FV, here nodes represent cell centers)
x = np.linspace(0, L, N)

# Number of unknowns (excluding the two boundary nodes)
n_unknown = N - 2

# Set up coefficient matrix A and right-hand side vector b for interior nodes
# Finite volume discretization using central differences results in:
# (-phi_{i-1} + 2*phi_i - phi_{i+1})*(Gamma/dx) = 0  =>  -phi_{i-1} + 2*phi_i - phi_{i+1} = 0
A = np.zeros((n_unknown, n_unknown))
b = np.zeros(n_unknown)

# Fill the matrix A and vector b
for i in range(n_unknown):
    A[i, i] = 2.0
    if i - 1 >= 0:
        A[i, i-1] = -1.0
    if i + 1 < n_unknown:
        A[i, i+1] = -1.0

# Incorporate Dirichlet boundary conditions into b
# For the first unknown (i=1 in the full array), phi_0 is known:
b[0] += phi_left
# For the last unknown (i=N-2 in full array), phi_{N-1} is known:
b[-1] += phi_right

# Solve the linear system for interior phi values
phi_interior = np.linalg.solve(A, b)

# Assemble the full phi solution including boundary conditions
phi = np.zeros(N)
phi[0] = phi_left
phi[-1] = phi_right
phi[1:-1] = phi_interior

# Save the final solution as a 1D numpy array to "phi.npy"
np.save('/PDE_Benchmark/results/prediction/o3-mini/prompts/phi_1D_Species_Diffusion.npy', phi)
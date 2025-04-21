#!/usr/bin/env python3
import numpy as np

# Parameters
Gamma = 1e-4            # diffusion coefficient [m^2/s]
phi_left = 10           # Dirichlet BC at x = 0
phi_right = 100         # Dirichlet BC at x = 0.1
L = 0.1                 # Domain length [m]
N = 100                 # Number of internal control volumes

# Calculate grid spacing (including boundary nodes)
dx = L / (N + 1)

# Assemble linear system for internal nodes (i = 1, ..., N)
# Finite Volume discretization gives:
#   -phi[i-1] + 2*phi[i] - phi[i+1] = 0
# with modification at the boundaries:
#   For i = 1: 2*phi[1] - phi[2] = phi_left
#   For i = N: -phi[N-1] + 2*phi[N] = phi_right

A = np.zeros((N, N))
b = np.zeros(N)

for i in range(N):
    A[i, i] = 2.0
    if i > 0:
        A[i, i - 1] = -1.0
    if i < N - 1:
        A[i, i + 1] = -1.0

# Incorporate boundary conditions into the right-hand side
b[0] = phi_left
b[-1] = phi_right

# Solve the linear system for internal phi values
phi_internal = np.linalg.solve(A, b)

# Assemble the full solution including boundary nodes
phi = np.empty(N + 2)
phi[0] = phi_left
phi[1:-1] = phi_internal
phi[-1] = phi_right

# Save the final solution (1D array) to phi.npy
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/phi_1D_Species_Diffusion.npy', phi)
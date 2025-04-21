import numpy as np

# Problem parameters
L = 0.1  # Domain length [m]
Gamma = 1e-4  # Diffusion coefficient [m^2/s]

# Discretization
nx = 100  # Number of control volumes
dx = L / (nx - 1)  # Grid spacing

# Grid
x = np.linspace(0, L, nx)

# Boundary conditions
phi_left = 10
phi_right = 100

# Coefficient matrix and source vector
A = np.zeros((nx, nx))
b = np.zeros(nx)

# Finite volume discretization
for i in range(1, nx-1):
    A[i, i-1] = -Gamma/dx
    A[i, i] = Gamma/dx + Gamma/dx
    A[i, i+1] = -Gamma/dx

# Apply boundary conditions
A[0, 0] = 1
b[0] = phi_left
A[-1, -1] = 1
b[-1] = phi_right

# Solve linear system
phi = np.linalg.solve(A, b)

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/phi_1D_Species_Diffusion.npy', phi)
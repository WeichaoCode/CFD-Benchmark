import numpy as np

# Problem parameters
Gamma = 1e-4
L = 0.1
phi_0 = 10
phi_L = 100
Nx = 100  # Number of cells

# Grid generation
dx = L / Nx
x = np.linspace(dx/2, L-dx/2, Nx)

# Initialize solution
phi = np.zeros(Nx)

# Discretization
A = np.zeros((Nx, Nx))
b = np.zeros(Nx)

# Interior nodes
for i in range(1, Nx - 1):
    A[i, i-1] = -Gamma / dx**2
    A[i, i] = 2 * Gamma / dx**2
    A[i, i+1] = -Gamma / dx**2

# Boundary conditions
A[0, 0] = 2 * Gamma / dx**2
A[0, 1] = -Gamma / dx**2
b[0] = Gamma / dx**2 * phi_0

A[Nx-1, Nx-2] = -Gamma / dx**2
A[Nx-1, Nx-1] = 2 * Gamma / dx**2
b[Nx-1] = Gamma / dx**2 * phi_L

# Solve the linear system
phi = np.linalg.solve(A, b)

# Save the solution
save_values = ['phi']
np.save('/PDE_Benchmark/results/prediction/gemini/prompts/phi_1D_Species_Diffusion.npy', phi)
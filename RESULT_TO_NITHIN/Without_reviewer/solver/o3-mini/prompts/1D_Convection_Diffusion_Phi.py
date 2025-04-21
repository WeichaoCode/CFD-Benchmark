import numpy as np

# Parameters
rho = 1.0            # kg/m^3
u = 2.5              # m/s
Gamma = 0.1          # kg/(m·s)
L = 1.0              # domain length
N = 5                # number of control volumes
dx = L / N         # cell width

# Coefficients for upwind finite volume discretization
F = rho*u          # Convective flux
D = Gamma/dx       # Diffusive conductance

aW = F + D
aE = D
aP = aW + aE

# Set up linear system Ax = b for unknown phi at cell centers.
A = np.zeros((N, N))
b = np.zeros(N)

# Boundary values (Dirichlet BCs)
phi_left = 1.0
phi_right = 0.0

# Cell 0 (first control volume)
A[0, 0] = aP
A[0, 1] = -aE
b[0] = aW * phi_left

# Interior cells
for i in range(1, N-1):
    A[i, i-1] = -aW
    A[i, i] = aP
    A[i, i+1] = -aE
    b[i] = 0.0

# Cell N-1 (last control volume)
A[N-1, N-2] = -aW
A[N-1, N-1] = aP
b[N-1] = aE * phi_right

# Solve the linear system
phi = np.linalg.solve(A, b)

# Save the final solution as a 1D numpy array in 'phi.npy'
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/phi_1D_Convection_Diffusion_Phi.npy', phi)
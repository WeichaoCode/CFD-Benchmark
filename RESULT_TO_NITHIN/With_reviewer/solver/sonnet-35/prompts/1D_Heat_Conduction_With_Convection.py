import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la

# Problem parameters
k = 1000.0  # Thermal conductivity [W/(m·K)]
h = 62.5    # Convective heat transfer coefficient [W/(m²·K)]
T_inf = 20.0  # Ambient temperature [°C]
hP_kA = 25.0  # Convective term coefficient [m⁻²]
L = 0.5     # Domain length [m]

# Discretization
nx = 100  # Number of grid points
dx = L / (nx - 1)  # Grid spacing

# Grid
x = np.linspace(0, L, nx)

# Boundary conditions
T_left = 100.0   # Left boundary temperature [°C]
T_right = 200.0  # Right boundary temperature [°C]

# Finite Volume Method
# Construct sparse matrix A and vector b
A = sp.lil_matrix((nx, nx))
b = np.zeros(nx)

# Interior points
for i in range(1, nx-1):
    # Diffusion term coefficients
    D_west = k / dx
    D_east = k / dx
    
    # Convection term
    S_u = hP_kA * T_inf * dx
    S_p = -hP_kA * dx
    
    # Assemble matrix
    A[i, i-1] = -D_west
    A[i, i] = D_west + D_east - S_p
    A[i, i+1] = -D_east
    b[i] = S_u

# Boundary conditions
A[0, 0] = 1.0
b[0] = T_left

A[-1, -1] = 1.0
b[-1] = T_right

# Convert to CSR format for solving
A_csr = A.tocsr()

# Solve linear system
T = la.spsolve(A_csr, b)

# Save solution
np.save('/PDE_Benchmark/results/prediction/sonnet-35/prompts/T_1D_Heat_Conduction_With_Convection.npy', T)
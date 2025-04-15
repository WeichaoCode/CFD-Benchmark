import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

# Domain parameters
Lx, Ly = 2, 1
Nx, Ny = 100, 50
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

# Grid generation
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Source term initialization
b = np.zeros((Ny, Nx))
b[int(Ny/4), int(Nx/4)] = 100
b[int(3*Ny/4), int(3*Nx/4)] = -100

# Construct matrix A and RHS vector
def create_poisson_matrix(Nx, Ny, dx, dy):
    # Coefficient matrix for 2D Poisson equation
    main_diag = np.zeros(Nx*Ny)
    lower_diag = np.zeros(Nx*Ny-1)
    upper_diag = np.zeros(Nx*Ny-1)
    lower_diag_x = np.zeros(Nx*Ny-Nx)
    upper_diag_x = np.zeros(Nx*Ny-Nx)

    for j in range(Ny):
        for i in range(Nx):
            k = j*Nx + i
            if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
                # Boundary points
                main_diag[k] = 1
            else:
                # Interior points
                main_diag[k] = -2*(1/dx**2 + 1/dy**2)
                if i > 0:
                    lower_diag[k-1] = 1/dx**2
                if i < Nx-1:
                    upper_diag[k] = 1/dx**2
                if j > 0:
                    lower_diag_x[k-Nx] = 1/dy**2
                if j < Ny-1:
                    upper_diag_x[k] = 1/dy**2

    # Create sparse matrix
    diagonals = [main_diag, lower_diag, upper_diag, lower_diag_x, upper_diag_x]
    offsets = [0, -1, 1, -Nx, Nx]
    A = diags(diagonals, offsets, shape=(Nx*Ny, Nx*Ny)).tocsr()
    return A

# Create RHS vector with boundary conditions
def create_rhs(Nx, Ny, b):
    rhs = b.flatten()
    # Apply Dirichlet boundary conditions
    for j in range(Ny):
        for i in range(Nx):
            k = j*Nx + i
            if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
                rhs[k] = 0
    return rhs

# Solve Poisson equation
A = create_poisson_matrix(Nx, Ny, dx, dy)
rhs = create_rhs(Nx, Ny, b)
p = spsolve(A, rhs).reshape((Ny, Nx))

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/haiku/prompts/p_2D_Poisson_Equation.npy', p)
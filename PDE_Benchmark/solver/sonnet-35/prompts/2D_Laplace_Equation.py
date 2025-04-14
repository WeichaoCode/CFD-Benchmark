import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Domain parameters
Lx, Ly = 2.0, 1.0
nx, ny = 100, 50  # Grid resolution
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialize potential field
p = np.zeros((ny, nx))

# Construct sparse matrix for Laplacian
def create_laplacian_matrix(nx, ny, dx, dy):
    # Create sparse matrix for 2D Poisson equation
    diagonals_x = np.zeros((3, nx*ny))
    diagonals_y = np.zeros((3, nx*ny))
    
    # X-direction coefficients
    diagonals_x[0, nx:] = 1 / (dx**2)  # Lower diagonal
    diagonals_x[1, :] = -2 / (dx**2)   # Main diagonal
    diagonals_x[2, :-nx] = 1 / (dx**2) # Upper diagonal
    
    # Y-direction coefficients
    diagonals_y[0, 1:] = 1 / (dy**2)   # Lower diagonal
    diagonals_y[1, :] = -2 / (dy**2)   # Main diagonal
    diagonals_y[2, :-1] = 1 / (dy**2)  # Upper diagonal
    
    # Combine x and y direction coefficients
    A = sp.diags(diagonals_x, [-nx, 0, nx], shape=(nx*ny, nx*ny))
    B = sp.diags(diagonals_y, [-1, 0, 1], shape=(nx*ny, nx*ny))
    
    return (A + B).tocsr()  # Convert to CSR format

# Apply boundary conditions
def apply_boundary_conditions(p, x, y):
    # Left boundary (x=0): p = 0
    p[:, 0] = 0
    
    # Right boundary (x=2): p = y
    p[:, -1] = y
    
    # Top and bottom boundaries: Neumann condition (zero gradient)
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]
    
    return p

# Solve Poisson equation using sparse matrix method
A = create_laplacian_matrix(nx, ny, dx, dy)
b = np.zeros(nx*ny)

# Solve linear system
p_flat = spla.spsolve(A, b)
p = p_flat.reshape((ny, nx))

# Apply boundary conditions
p = apply_boundary_conditions(p, x, y)

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/p_2D_Laplace_Equation.npy', p)
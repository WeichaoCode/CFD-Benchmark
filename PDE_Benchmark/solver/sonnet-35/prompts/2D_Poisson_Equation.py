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

# Source term
b = np.zeros((ny, nx))
b[int(ny/4), int(nx/4)] = 100
b[int(3*ny/4), int(3*nx/4)] = -100

# Construct sparse matrix for Poisson equation
def poisson_matrix(nx, ny, dx, dy):
    # Create sparse matrix using COO format for easier modification
    rows, cols, data = [], [], []
    
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            
            # Diagonal term
            rows.append(idx)
            cols.append(idx)
            data.append(-2 * (1/dx**2 + 1/dy**2))
            
            # X-direction neighbors
            if i > 0:
                rows.append(idx)
                cols.append(idx-1)
                data.append(1/dx**2)
            
            if i < nx-1:
                rows.append(idx)
                cols.append(idx+1)
                data.append(1/dx**2)
            
            # Y-direction neighbors
            if j > 0:
                rows.append(idx)
                cols.append(idx-nx)
                data.append(1/dy**2)
            
            if j < ny-1:
                rows.append(idx)
                cols.append(idx+nx)
                data.append(1/dy**2)
    
    # Create sparse matrix
    A = sp.csr_matrix((data, (rows, cols)), shape=(nx*ny, nx*ny))
    return A

# Apply boundary conditions
def apply_boundary_conditions(A, b, nx, ny):
    # Convert to LIL format for easier modification
    A = A.tolil()
    
    # Zero Dirichlet on all boundaries
    for i in range(nx):
        # Bottom boundary
        idx = i
        A[idx, :] = 0
        A[idx, idx] = 1
        b[idx] = 0
        
        # Top boundary
        idx = (ny-1)*nx + i
        A[idx, :] = 0
        A[idx, idx] = 1
        b[idx] = 0
    
    for j in range(ny):
        # Left boundary
        idx = j*nx
        A[idx, :] = 0
        A[idx, idx] = 1
        b[idx] = 0
        
        # Right boundary
        idx = j*nx + (nx-1)
        A[idx, :] = 0
        A[idx, idx] = 1
        b[idx] = 0
    
    # Convert back to CSR for solving
    return A.tocsr(), b

# Solve Poisson equation
A = poisson_matrix(nx, ny, dx, dy)
b_flat = b.flatten()
A, b_flat = apply_boundary_conditions(A, b_flat, nx, ny)

# Solve linear system
p_flat = spla.spsolve(A, b_flat)
p = p_flat.reshape((ny, nx))

# Save solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/sonnet-35/prompts/p_2D_Poisson_Equation.npy', p)
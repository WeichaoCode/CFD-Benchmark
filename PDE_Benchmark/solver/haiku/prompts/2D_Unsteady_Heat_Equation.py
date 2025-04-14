import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Problem parameters
Lx, Ly = 1.0, 1.0  # Domain size
nx, ny = 100, 100  # Grid points
alpha = 1.0  # Diffusivity
Q0 = 200.0  # Source magnitude
sigma = 0.1  # Source width
t_start, t_end = 0.0, 3.0  # Time domain

# Grid generation
dx = 2*Lx / (nx-1)
dy = 2*Ly / (ny-1)
x = np.linspace(-Lx, Lx, nx)
y = np.linspace(-Ly, Ly, ny)
X, Y = np.meshgrid(x, y)

# Time stepping
nt = 500
dt = (t_end - t_start) / nt

# Initial condition
T = 1.0 + Q0 * np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Precompute source term
def source(x, y, t):
    return Q0 * np.exp(-(x**2 + y**2) / (2 * sigma**2))

# Finite difference matrix construction
def build_matrix(nx, ny, dx, dy, dt, alpha):
    # Central difference coefficients
    cx = alpha * dt / (dx**2)
    cy = alpha * dt / (dy**2)
    
    # Matrix size
    N = nx * ny
    
    # Diagonal entries
    diags = np.zeros((5, N))
    diags[2, :] = 1 + 2*cx + 2*cy  # Main diagonal
    
    # Off-diagonal entries for x-direction
    diags[1, nx:] = -cx  # Lower diagonal
    diags[3, :-nx] = -cx  # Upper diagonal
    
    # Off-diagonal entries for y-direction
    diags[0, 1::nx] = -cy  # Lower diagonal
    diags[4, ::nx] = -cy   # Upper diagonal
    
    # Construct sparse matrix in CSR format
    offsets = [-1, -nx, 0, nx, 1]
    A = sp.spdiags(diags, offsets, N, N).tocsr()
    
    return A

# Time-stepping loop
A = build_matrix(nx, ny, dx, dy, dt, alpha)

for n in range(nt):
    t = t_start + n*dt
    
    # Source term
    Q = source(X, Y, t)
    
    # Right-hand side
    b = T.flatten() + dt * Q.flatten()
    
    # Enforce boundary conditions
    b[0:nx] = 1.0  # Bottom boundary
    b[-nx:] = 1.0  # Top boundary
    b[::nx] = 1.0  # Left boundary
    b[nx-1::nx] = 1.0  # Right boundary
    
    # Solve linear system
    T_new = spla.spsolve(A, b).reshape((ny, nx))
    
    # Update solution
    T = T_new

# Save final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/haiku/prompts/T_2D_Unsteady_Heat_Equation.npy', T)
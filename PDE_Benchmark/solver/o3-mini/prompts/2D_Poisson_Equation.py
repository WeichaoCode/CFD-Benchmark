import numpy as np

# Domain parameters
Lx = 2.0
Ly = 1.0
nx = 101  # number of grid points in x
ny = 51   # number of grid points in y
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Initialize p and b
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Set source terms
# Find indices corresponding to the source locations
# Source 1: b = 100 at x = Lx/4 and y = Ly/4
x1 = Lx / 4
y1 = Ly / 4
i1 = np.argmin(np.abs(x - x1))
j1 = np.argmin(np.abs(y - y1))
b[j1, i1] = 100.0

# Source 2: b = -100 at x = 3*Lx/4 and y = 3*Ly/4
x2 = 3 * Lx / 4
y2 = 3 * Ly / 4
i2 = np.argmin(np.abs(x - x2))
j2 = np.argmin(np.abs(y - y2))
b[j2, i2] = -100.0

# Iterative solver parameters (SOR)
max_iter = 10000
tol = 1e-6
omega = 1.5

# Precompute coefficients
dx2 = dx * dx
dy2 = dy * dy
denom = 2 * (dx2 + dy2)

for it in range(max_iter):
    p_old = p.copy()
    # Update interior points
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            r = ((p[j, i+1] + p[j, i-1]) * dy2 +
                 (p[j+1, i] + p[j-1, i]) * dx2 -
                 b[j, i] * dx2 * dy2)
            p_new = r / denom
            # SOR relaxation update
            p[j, i] = (1 - omega) * p[j, i] + omega * p_new
    
    # Enforce boundary conditions: Dirichlet p=0 at boundaries
    p[0, :]   = 0  # y = 0
    p[-1, :]  = 0  # y = Ly
    p[:, 0]   = 0  # x = 0
    p[:, -1]  = 0  # x = Lx

    # Check for convergence
    error = np.linalg.norm(p - p_old, ord=np.inf)
    if error < tol:
        break

# Save the final solution as a 2D numpy array in p.npy
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/o3-mini/prompts/p_2D_Poisson_Equation.npy', p)
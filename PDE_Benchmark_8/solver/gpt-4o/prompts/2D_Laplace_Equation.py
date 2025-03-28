import numpy as np

# Domain parameters
Lx, Ly = 2.0, 1.0
nx, ny = 51, 51  # Number of grid points
dx, dy = Lx / (nx - 1), Ly / (ny - 1)

# Create grid
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Initialize potential field
p = np.zeros((ny, nx))

# Boundary conditions
p[:, 0] = 0  # Left boundary
p[:, -1] = y  # Right boundary

# Iterative solver parameters
tolerance = 1e-5
max_iterations = 10000

# Successive Over-Relaxation (SOR) method
omega = 1.5  # Relaxation factor

# Iterative solver
for iteration in range(max_iterations):
    p_old = p.copy()
    
    # Update interior points
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            p[j, i] = ((p[j, i+1] + p[j, i-1]) * dy**2 +
                       (p[j+1, i] + p[j-1, i]) * dx**2) / (2 * (dx**2 + dy**2))
            p[j, i] = omega * p[j, i] + (1 - omega) * p_old[j, i]
    
    # Neumann boundary conditions (top and bottom)
    p[0, :] = p[1, :]  # Bottom boundary
    p[-1, :] = p[-2, :]  # Top boundary
    
    # Check for convergence
    if np.linalg.norm(p - p_old, ord=np.inf) < tolerance:
        break

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts/p_2D_Laplace_Equation.npy', p)
import numpy as np

# Domain parameters
Lx, Ly = 2.0, 1.0
nx, ny = 101, 51
dx, dy = Lx / (nx - 1), Ly / (ny - 1)

# Initialize the potential field
p = np.zeros((ny, nx))

# Boundary conditions
# Left boundary (x = 0): p = 0
p[:, 0] = 0

# Right boundary (x = 2): p = y
p[:, -1] = np.linspace(0, Ly, ny)

# Top and bottom boundaries (y = 0, 1): Neumann condition ∂p/∂y = 0
# This is implicitly handled by not updating the first and last rows in the y-direction

# Iterative solver parameters
tolerance = 1e-5
max_iterations = 10000

# Iterative solver using the Gauss-Seidel method
for iteration in range(max_iterations):
    p_old = p.copy()
    
    # Update the interior points
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            p[j, i] = 0.5 * ((p[j, i+1] + p[j, i-1]) * dy**2 + (p[j+1, i] + p[j-1, i]) * dx**2) / (dx**2 + dy**2)
    
    # Check for convergence
    if np.linalg.norm(p - p_old, ord=np.inf) < tolerance:
        break

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gpt-4o/prompts/p_2D_Laplace_Equation.npy', p)
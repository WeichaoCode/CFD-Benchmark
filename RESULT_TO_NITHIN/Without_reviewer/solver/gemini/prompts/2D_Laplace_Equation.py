import numpy as np

# Parameters
nx = 50
ny = 25
dx = 2 / (nx - 1)
dy = 1 / (ny - 1)

# Domain
x = np.linspace(0, 2, nx)
y = np.linspace(0, 1, ny)

# Initialize p
p = np.zeros((ny, nx))

# Boundary conditions
p[:, 0] = 0  # Left boundary
p[:, -1] = y  # Right boundary

# Iteration parameters
max_iter = 10000
tolerance = 1e-6

# Solve using finite difference method
for iteration in range(max_iter):
    p_old = np.copy(p)

    # Update interior points
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            p[i, j] = 0.5 * ((p[i, j+1] + p[i, j-1]) * dy**2 + (p[i+1, j] + p[i-1, j]) * dx**2) / (dx**2 + dy**2)

    # Neumann boundary conditions (top and bottom)
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]

    # Check for convergence
    max_diff = np.max(np.abs(p - p_old))
    if max_diff < tolerance:
        break

# Save the solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark/results/prediction/gemini/prompts/p_2D_Laplace_Equation.npy', p)
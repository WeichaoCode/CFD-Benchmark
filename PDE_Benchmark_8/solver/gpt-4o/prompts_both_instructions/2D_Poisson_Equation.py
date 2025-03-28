import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
Lx, Ly = 2.0, 1.0
nx, ny = 50, 50
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Initialize p and b
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Source term
b[int(ny / 4), int(nx / 4)] = 100
b[int(3 * ny / 4), int(3 * nx / 4)] = -100

# Convergence parameters
tolerance = 1e-4
max_iterations = 10000

# Iterative solver (Jacobi method)
for iteration in range(max_iterations):
    p_old = p.copy()
    
    # Update p using the finite difference method
    p[1:-1, 1:-1] = ((p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dy**2 +
                     (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dx**2 -
                     b[1:-1, 1:-1] * dx**2 * dy**2) / (2 * (dx**2 + dy**2))
    
    # Apply Dirichlet boundary conditions
    p[0, :] = 0
    p[-1, :] = 0
    p[:, 0] = 0
    p[:, -1] = 0
    
    # Check for convergence
    if np.linalg.norm(p - p_old, ord=np.inf) < tolerance:
        print(f"Converged after {iteration} iterations")
        break

# Save the final pressure field
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/p_2D_Poisson_Equation.npy', p)

# Visualize the result
plt.figure(figsize=(8, 4))
plt.contourf(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), p, 50, cmap='jet')
plt.colorbar(label='Pressure')
plt.title('Pressure Field')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
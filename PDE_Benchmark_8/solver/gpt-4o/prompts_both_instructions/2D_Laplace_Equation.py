import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
nx, ny = 31, 31
dx = 2 / (nx - 1)
dy = 1 / (ny - 1)

# Initialize the potential field p
p = np.zeros((ny, nx))

# Boundary conditions
p[:, 0] = 0  # Left boundary (x = 0)
p[:, -1] = np.linspace(0, 1, ny)  # Right boundary (x = 2)

# Convergence criteria
tolerance = 1e-5
max_iterations = 10000

# Iterative solver using Gauss-Seidel method
for iteration in range(max_iterations):
    p_old = p.copy()
    
    # Update the interior points
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            p[j, i] = ((dy**2 * (p[j, i+1] + p[j, i-1]) +
                        dx**2 * (p[j+1, i] + p[j-1, i])) /
                       (2 * (dx**2 + dy**2)))
    
    # Neumann boundary conditions (top and bottom)
    p[0, :] = p[1, :]  # Bottom boundary (y = 0)
    p[-1, :] = p[-2, :]  # Top boundary (y = 1)
    
    # Check for convergence
    max_diff = np.max(np.abs(p - p_old))
    if max_diff < tolerance:
        print(f"Converged after {iteration} iterations.")
        break
else:
    print("Did not converge within the maximum number of iterations.")

# Save the final solution
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_both_instructions/p_2D_Laplace_Equation.npy', p)

# Visualization
plt.contourf(np.linspace(0, 2, nx), np.linspace(0, 1, ny), p, 20, cmap='viridis')
plt.colorbar(label='Potential p')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour plot of the potential field p(x, y)')
plt.show()
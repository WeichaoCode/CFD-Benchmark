import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 31, 31
dx = 2 / (nx - 1)
dy = 1 / (ny - 1)
tolerance = 1e-5

# Initialize p with zeros
p = np.zeros((ny, nx))

# Iterative solver
def solve_poisson(p, dx, dy, tolerance):
    max_diff = tolerance + 1
    while max_diff > tolerance:
        p_old = p.copy()
        
        # Update interior points
        p[1:-1, 1:-1] = ((p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dy**2 +
                         (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dx**2) / (2 * (dx**2 + dy**2))
        
        # Apply boundary conditions
        p[:, 0] = 0  # Left boundary
        p[:, -1] = np.linspace(0, 1, ny)  # Right boundary
        p[0, :] = p[1, :]  # Bottom boundary (Neumann)
        p[-1, :] = p[-2, :]  # Top boundary (Neumann)
        
        # Calculate the maximum difference for convergence check
        max_diff = np.max(np.abs(p - p_old))
    
    return p

# Solve the PDE
p_final = solve_poisson(p, dx, dy, tolerance)

# Save the final solution to a .npy file
np.save('/opt/CFD-Benchmark/PDE_Benchmark_8/results/prediction/gpt-4o/prompts_instruction_2/p_final_2D_Laplace_Equation.npy', p_final)

# Visualization
X, Y = np.meshgrid(np.linspace(0, 2, nx), np.linspace(0, 1, ny))
plt.contourf(X, Y, p_final, 20, cmap='viridis')
plt.colorbar(label='Potential p(x, y)')
plt.title('Contour plot of the potential field')
plt.xlabel('x')
plt.ylabel('y')
plt.show()